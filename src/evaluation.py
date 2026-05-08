import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
import json

def mean_mae(y_true_test : np.ndarray,y_pred : np.ndarray):
    '''
    Calculating a mean MAE for the given arrays
    '''
    return np.mean(np.abs((y_true_test-y_pred)))

def rmse(y_true_test : np.ndarray, y_pred : np.ndarray):
    return np.sqrt(mean_squared_error(y_true_test , y_pred))

def make_df_results(y_true_test : np.ndarray, y_pred : np.ndarray, timestamps : np.ndarray):
    df_eval = pd.DataFrame({
    'time': timestamps,
    'actual': y_true_test,
    'predicted': y_pred
    })

    return df_eval

def create_result_folder():
    os.makedirs("results", exist_ok=True)

def save_json(path: Path, data: dict):
    """save json data at the path
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"json file saved at: {path}")    

def save_score(mae_score, rmse_score):
    scores = {"mae": mae_score, "rmse": rmse_score}
    save_json(path=Path("results/scores.json"), data=scores)

def save_hyperparams(alpha, l1_ratio):
    scores = {"alpha": alpha, "l1_ratio": l1_ratio}
    save_json(path=Path("hyperparams.json"), data=scores)

def save_plot_predicted(y_true_test : np.ndarray ,y_pred : np.ndarray, timestamps : np.ndarray):
    df_eval = make_df_results(y_true_test,y_pred, timestamps)


    plt.figure(figsize=(12,5))

    plt.plot(df_eval["time"].values,df_eval["actual"].values, label="Actual", linewidth=2)
    plt.plot(df_eval["time"].values,df_eval["predicted"].values, label="Predicted", alpha=1)


    plt.legend()
    plt.title("Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig("results/plot_predicted.png")
    print(f"Plot of predictions saved at: results/plot_predicted.png") 

def save_plot_residuals(y_true_test: np.ndarray, y_pred : np.ndarray, timestamps : np.ndarray):
    residuals = y_pred - y_true_test
    df_eval = make_df_results(y_true_test,y_pred, timestamps)

    plt.figure(figsize=(12,4))
    plt.plot(df_eval["time"].values,residuals)
    plt.axhline(0, color='red', linestyle='--')

    plt.title("Residuals (Prediction - Actual)")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.savefig("results/plot_residuals.png")    
    print(f"Plot of residuals saved at: results/plot_residuals.png") 