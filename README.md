# Hungarian Day-Ahead Electricity Price Forecasting

## 1. Project Overview

This pipeline produces 24-hour-ahead point forecasts of Hungarian day-ahead electricity prices (HUPX), one **ElasticNet** model per delivery hour, using the three provided exogenous features alongside engineered calendar and lag features.

ElasticNet was chosen for four reasons:

- **Data efficiency.** A regularised linear model generalises well on a few years of hourly data where tree-based or neural models would overfit or require heavy tuning.
- **Interpretability.** Coefficients can be inspected directly to verify the model is learning possible good relationships.
- **Speed.** 24 models retrained daily in a walk-forward loop complete very fast.
- **Structure fits the model.** Strong regular seasonality which can be encoded as linear features — intraday shape, weekday effects and etc.

---

## 2. Quickstart

```bash
conda create --name <env_name> python=3.11
conda activate <env_name>

pip install -r requirements.txt

python main.py
```

---

## 3. Data & Feature Engineering

### DST Handling

Hungarian electricity data follows CET, which creates irregularities at DST transitions: 23-hour days (spring) produce a missing hour that was **linearly interpolated**; 25-hour days (autumn) produce a duplicate hour where the **second occurrence was discarded**. This ensures a clean, evenly-spaced hourly index throughout.

### Engineered Features


**Demand-solar interaction** — `demand_minus_solar` captures the net load that must be served by dispatchable generation, since solar directly offsets demand. 

**Cross-feature interactions** — pairwise products of all exogenous features 
(`demand × solar`, `demand × net_exchange`, `solar × net_exchange`). These encode 
non-linear relationships that a linear model cannot discover from individual features 
alone.

**Ramp features** — differences between current and lagged values for exogenous features 
(`curr - lag_24`, `curr - lag_168`), and between consecutive lags for price 
(`lag_24 - lag_48`, `lag_48 - lag_168`). These capture the rate of change rather than 
the absolute level — a sudden drop in solar or spike in net export can be more informative 
than the level alone.

**Calendar features** — day of week and season. Intraday and weekday price shapes are 
structurally different and encode cleanly as linear features.

**Fourier features** — sine/cosine pairs over the monthly cycle to represent smooth 
seasonal variation without requiring the model to infer it from calendar dummies.

**Lags** — 24h, 48h, and 168h lags for the target and all exogenous features including 
`demand_minus_solar`. Horizons align with the same hour on the previous day, two days 
prior, and the same weekday last week. All lag choices are respected to 
prevent leakage.

**Rolling statistics** — daily mean, std, min, max over 24h and 7d windows. Same-hour-7d 
mean and std additionally capture weekday-specific level and volatility. Applied to price 
only.

**Cross-temporal structure** — raw hourly data is reshaped to `(days, features × 24)` and the target to `(days,  24)`.
before modelling, producing one wide row per day containing all 24 hours' feature values 
concatenated. This means each model implicitly sees the full intraday picture — hour 13's 
model has direct access to hour 12's and hour 14's lag, ramp, and exogenous values as 
input columns, without any explicit feature engineering. Each of the 24 ElasticNet models 
is trained on this same wide matrix but predicts only its own delivery hour, preserving 
per-hour specialisation while retaining cross-hour information.




### Feature Selection

Although ElasticNet already performs implicit shrinkage and sparsity selection, 
the reshaped daily-wide representation `(days, features × 24)` substantially increases 
the dimensionality of the input space. An explicit feature selection stage therefore 
helps remove noisy and redundant inputs before final fitting, improving stability and 
reducing the risk of overfitting in the high-dimensional setting.

That is why, a two-step selection is applied to keep only the most informative 
features and reduce noise in the input space.

**Step 1 — Correlation filtering** (global, applied once on training data before fitting): feature pairs with correlation above 0.95 are identified and the lower-variance feature is 
dropped. ElasticNet's L2 penalty handles multicollinearity well, but removing 
near-duplicate features reduces redundancy before any model sees the data.

**Step 2 — SelectFromModel** (per hour, applied inside the walk-forward loop): for each 
of the 24 delivery hours, a preliminary ElasticNet is fit and only features whose 
coefficient magnitude exceeds a threshold are retained(0.001). The final ElasticNet for that 
hour is then trained on this reduced feature set. This means each hour's model performs 
its own independent selection — hour 3 (overnight load) and hour 13 (solar peak) 
are likely to retain structurally different feature subsets, which is the correct 
behaviour given their different price dynamics, rather than static features for each hour.

Both steps use training data only to prevent leakage into validation and test periods.

### Input Scaling

All features scaled with **RobustScaler** due to extreme spike values might otherwise dominate ElasticNet's regularisation.

### Missing Values

NaNs introduced by lagging and rolling were dropped. These fall exclusively in the first week of the dataset — old, sparse, and negligible relative to total data size.

### Target Transformation

Target was scaled with **RobustScaler** (resistant to price spikes), then transformed with **arcsinh** — equivalent to log for large positive values but well-defined for negative prices. [Chęć, Uniejewski & Weron – Electricity price seasonality forecasting paper (WUST).]

### Reshaping

Raw data is hourly `(timesteps × features)`. Before modelling, it is reshaped to `(days × features)`, where each of the 24 ElasticNet models sees one row per day and is trained and evaluated on it.

---

## 4. Model Choice & Reasoning

The core model is **ElasticNet** — a regularised linear regression with both L1 and L2 penalties — making this an **autoregressive model with exogenous features**, where past prices and external signals enter as engineered lag and rolling features rather than through a dedicated AR structure.

**Why ElasticNet:**

- **Structure fits the model.** Seasonality is already explicitly encoded through Fourier terms, calendar features, and same-hour lags — leaving the model to learn residual linear relationships rather than discover structure from scratch.
- **24 separate models.** One model per delivery hour means each model specialises in its hour's dynamics — midday solar suppression, morning ramp, overnight baseload. This cleanly reduces variance compared to a single model handling all 24 hours at once.
- **Interpretability, speed, and data efficiency.** Coefficients are directly inspectable, 24 models retrain in seconds, and regularisation handles the moderate dataset size (~731 days) without overfitting.

**Alternatives considered:**

| Model | Outcome |
|---|---|
| **Seasonal naive** (same hour 7 days prior) | Baseline. Useful floor to beat. |
| **LightGBM** | Strong on tabular data in general, but  was worse than ElasticNet here |

Neural network were not even considered due to to computational cost, time and lack of data(24 outputs -> 731 around point to train). Instead, cross temporal and cross-feature interactions were introduced to our models, so they can benefit and mimic kind of neural net behaviour.

Rather than switching to a neural architecture, cross-temporal and cross-feature 
interactions were introduced through feature engineering — the reshaped `(days × 
features×24)` input matrix gives each ElasticNet model access to the full intraday 
picture, recovering some of the interaction modelling that a neural network might learn 
implicitly. ElasticNet outperforming LightGBM on well-engineered features is consistent 
with the electricity price forecasting literature, where regularised linear models remain 
competitive even against tree-based methods. [Chęć, Uniejewski & Weron – Electricity 
price forecasting, WUST.]

---

## 5. Validation Strategy

The dataset is split into three consecutive temporal blocks:

```
|——————— Train ———————|——— Validation (45 days) ———|——— Test (45 days) ———|
```

**Why 45 days?** Large enough to cover stable price conditions across several weeks and capture weekday/weekend cycles reliably, but short enough to avoid spanning a full season.

**Why not random split?** Electricity prices are sequentially correlated and a random split does not respect the time axis strictly.

**Walk-forward evaluation** is used for both validation and test periods. On each day *d*:
Split 1:
|------ TRAIN ------| TEST day 1 |

Split 2:
|-------- TRAIN --------| TEST day 2 |

Split 3:
|----------- TRAIN -----------| TEST day 3 |

1. Retrain on all data up to day *d*
2. Forecast the 24 hours of day *d+1*
3. Record errors, retrain all data up to day *d+1*, forecast *d+2* and etc.

This expanding window mirrors production well. The model sees more data as time progresses, and yesterday's realised price feeds into today's lag features. Retraining daily is important precisely because the 24h and 48h lags can make the most recent observation highly influential. Also forecasting daily is usual.

**Validation vs test distinction** — hyperparameters (ElasticNet α and l1-ratio) should be tuned based on validation period errors only. The test period should be touched once, at the very end, to report the final score. 

All transformations, feature selection steps, and scaling operations were fit exclusively 
on training data inside each walk-forward iteration to prevent temporal leakage.

---

## 6. Results & Error Metrics

**Metrics used:** MAE and RMSE. A lot of papers use these, with MAE being main one, since it shows the average absolute forecasting error in €/MWh and treats all deviations equally as well as easy to interpret since absolute values. RMSE complements MAE by penalising large errors more heavily, making it sensitive to price spikes, which shows how the model performing under extreme conditions. For relative metric, to see in percenteges, usually MAPE used,however, for prices in electricy market, MAPE was excluded because it behaves poorly when true values are near or at zero, which occurs quie frequent in this dataset .

| Model | MAE | RMSE |
|---|---|---|
| Baseline | `24.93` | `41.17` |
| LightGBM | `17.56` | `27.69` |
| **24 Models: Elastic Net** | `14.37` | `24.26` |

Given the price range of roughly €60–370/MWh, the achieved MAE represents a reasonable forecast error — the model captures the dominant patterns without huge deviations u.

**Plot — Predicted vs Actual (Test Period)**

The plot reveals several characteristics of model behaviour:
![alt text](results/plot_predicted.png)

- **Main patterns are well captured** — intraday shape, weekday structure, and general price level track closely throughout most of the test period except for the spike.
- **One spike event dominates the error.** Removing this single episode, MAE would likely fall to around 7–8 €/MWh.

**Plot — Residuals (Test Period)**

![alt text](results/plot_residuals.png)

- **Residuals are approximately zero-centred** — indicating almost no systematic bias in either direction.
- **Model stabilises after the spike** — suggesting the expanding window absorbs the shock and recovers, rather than remaining persistently miscalibrated.


---

## 7. Limitations & Failure Modes

**Spike events.** The model struggles with sharp, short-timewise price spikes — as visible in the test period around November. After some readingthis coincided with an unusually cold period, where heating demand probably pushed prices far outside the normal range. 

**Missing features.** The provided feature set covers demand and solar, but European electricity prices are also strongly driven by Wind generation, Gas prices, Tempreature, Weather.Adding these can have a high impact on the current model.

**Data size.** ~730 days is modest. The ElasticNet handles this well, but limits the ability to model rare regimes  that appear only a handful of times in the training history.  Although cross-temporal and cross-feature interactions were introduced to partially 
approximate the type of dependencies typically learned by neural networks, they are 
a simplified. With additional years of data,  more expressive models such as LSTMs — which have demonstrated strong performance in electricity price forecasting can potentially yield further improvements due to their ability to capture complex non-linear temporal dependencies and higher-order feature interactions more effectively than linear models.

**Limited capability to adapt to sturctual changes.** While there is walk-forward retrainig and some dynamic feature selection, model cannot detect drifts explicitly and respond to it accordingly. 

---

## 8. Code Structure

```
src/
  data.py         # data loading, converting to CET, DST handling
  features.py     # feature engineering, scaling, target transformation, splitting
  model.py        # defines train/validation and test sizes, walk-forward loop: expanding window retraining, TimeSeriesSplit for hyperparameter tuning and evaluation
  evaluate.py     # metrics, plots
main.py           # calls src modules in order
requirements.txt      # dependencies
results/              # scores/plots
```
