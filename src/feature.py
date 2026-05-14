import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def demand_minus_solar(df : pd.DataFrame):
  '''
  creating new feature by susbtrating solar forect generation from demand forecast
  '''
  df = df.copy()
  df['demand_minus_solar'] = df['demand_forecast_mwh'] - df['solar_forecast_mwh']
  return df

def calc_ramps(df, columns = ['demand_forecast_mwh','solar_forecast_mwh','net_exchange_forecast_mwh','price_eur_mwh']):
    df = df.copy()
    
    for col in columns:
        if col == 'price_eur_mwh':
            df[f'{col}_ramp_24_48'] = df[f'{col}_lag_24'] - df[f'{col}_lag_48']
            df[f'{col}_ramp_48_168'] = df[f'{col}_lag_48'] - df[f'{col}_lag_168']
        else:
            df[f'{col}_ramp_curr_24'] = df[col] - df[f'{col}_lag_24']
            df[f'{col}_ramp_curr_168'] = df[col] - df[f'{col}_lag_168']

    return df    

'''  def cross_intreactions(df,columns = ['demand_forecast_mwh','solar_forecast_mwh','net_exchange_forecast_mwh']):
    df = df.copy()
    for i in range(len(columns)):
        for j in range(i+1,len(columns)):
            df[f'{columns[i]} x {columns[j]}'] = df[columns[i]] * df[columns[j]]
    return df'''   

    
def get_calendar_features(df : pd.DataFrame):
  '''
  getting calendar features such as year, month, day, hour, a day of a week, is_weekend flag, and season
  '''
  df = df.copy()
  df['year'] = df.index.year
  df['month'] = df.index.month
  df['day'] = df.index.day
  df['hour'] = df.index.hour
  df['dayofweek'] = df.index.dayofweek


  def get_season(month):
      if month in [12,1,2]:
          return 0
      elif month in [3,4,5]:
          return 1
      elif month in [6,7,8]:
          return 2
      else:
          return 3
  df['season'] = df['month'].apply(get_season)
  df = pd.get_dummies(df, columns=['season'])
  df = pd.get_dummies(df, columns=['dayofweek'])


  return df

def drop_unused_features(df : pd.DataFrame,cols_to_drop : list =['day','hour','month','year']):
    '''
    Dropping unnecessary features and NaNs
    '''
    df = df.copy()
    df = df.drop(columns=cols_to_drop)
    df = df.dropna()
    return df


def add_lags(df : pd.DataFrame, columns : list = ['price_eur_mwh'], lags : list = [24,48,168]):
    """
    Adds lag(24h,48h,168h) features to a DataFrame based on the columns provided. 
    """
    df_lags = df.copy()
    for col in columns:

        for lag in lags:
            df_lags[f'{col}_lag_{lag}'] = df_lags[col].shift(lag)


    return df_lags

def add_fourier_features(df : pd.DataFrame,column_name : str, period : int, order : int = 4):
    '''
    sine/cosine pairs over the provided cycle. Foe example, monthly cycles. It can help smooth seasonal variation without requiring the model to infer it from calendar dummies.
    '''
    df=df.copy()
    
    for k in range(1, order + 1):
        df[f'fourier_{period}_sin_order_{k}'] = np.sin(2 * np.pi * k * df[column_name] / period)
        df[f'fourier_{period}_cos_order_{k}'] = np.cos(2 * np.pi * k * df[column_name] / period)
    return df   

def add_rolling_stats(df : pd.DataFrame, target_col : str = 'price_eur_mwh'):
    '''
    24h and 168h windows: mean, std, min, max. Same-hour-7d mean and std additionally capture weekday-specific level and volatility.
    Applied to a target column
    '''

    day = df[target_col].resample('D').agg(price_mean_24h = 'mean', price_std_24h = 'std',min_24h = 'min', max_24h = 'max').shift(1)
    day['price_mean_7d'] = df[target_col].resample('D').mean().rolling(7).mean().shift(1)
    day['price_std_7d'] = df[target_col].resample('D').mean().rolling(7).std().shift(1)
    day['price_max_7d'] = df[target_col].resample('D').max().rolling(7).max().shift(1)
    day['price_min_7d'] = df[target_col].resample('D').min().rolling(7).min().shift(1)
   

    df_rolling_stats = df.copy()
    df_rolling_stats['date'] = df_rolling_stats.index.date
    day.index = day.index.date

    df_rolling_stats= df_rolling_stats.join(day, on='date')
    df_rolling_stats.drop('date',axis=1,inplace=True)
    df_rolling_stats['same_hour_mean_7d'] = df.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(7).mean())
    df_rolling_stats['same_hour_std_7d'] = df.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(7).std())

    return df_rolling_stats

def scale_features(X_train : pd.DataFrame, X_test : pd.DataFrame):
    '''
    Robust Scaling features
    '''
    num_cols = X_train.select_dtypes(include=['number']).columns
    scaler = RobustScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test
        

def scale_target(y_train : pd.DataFrame):
    '''
    Robust Scaling target + asinh transformation
    '''
    scaler = RobustScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_train_scaled = np.arcsinh(y_train_scaled)
    return y_train_scaled,scaler

def inverse_scale_target(y_pred : np.ndarray, scaler : RobustScaler):
    '''
    Just inversing back to a normal scale
    '''
    y_pred = np.sinh(y_pred)
    y = scaler.inverse_transform(y_pred)
    return y
    

def corr_filtering(df, threshold = 0.95):
    X = df.drop('price_eur_mwh', axis=1)
  


    float_cols = X.select_dtypes(exclude=['bool']).columns
    bool_cols = X.select_dtypes(include=['bool']).columns
    
    X_train_bool = X[bool_cols]
    X_train = X[float_cols]
    
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    

    return to_drop
   


def cross_temp(df : pd.DataFrame, interactions = ['demand_forecast_mwh','solar_forecast_mwh','net_exchange_forecast_mwh','price_eur_mwh_lag_24','price_eur_mwh_lag_168'], n_neighbours = 3):
    '''
    Introducing cross-temporal and cross-feature interaction with default window 3: [h-3,h+3]
    '''
    df=df.copy()

    new_cols = {}
    window = n_neighbours
    
    for h in range(24):
        for offset in range(-window,window+1):

            neighbour_h = h + offset 
            if neighbour_h<0 or neighbour_h>23: 
                continue
            for f1 in range(len(interactions)):
                for f2 in range(f1 ,len(interactions)): 
                    if neighbour_h < h  or (offset==0 and f1==f2):
                        continue
                    new_cols[f'{interactions[f1]}_h{h:02d}x{interactions[f2]}_h{neighbour_h:02d}'] = df[f'{interactions[f1]}_h{h:02d}'] * df[f'{interactions[f2]}_h{neighbour_h:02d}']#
    return pd.concat([df,pd.DataFrame(new_cols)],axis=1)    