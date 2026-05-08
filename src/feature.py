from .data import reshape_df
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
  df['is_weekend'] = df.index.dayofweek >=5

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

def drop_unused_features(df : pd.DataFrame,cols_to_drop : list =['day','hour','month']):
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
        df[f'fourier_{period}_sin_order_{order}'] = np.sin(2 * np.pi * k * df[column_name] / period)
        df[f'fourier_{period}_cos_order_{order}'] = np.cos(2 * np.pi * k * df[column_name] / period)
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

def scale_features(X_train : np.ndarray, X_test : np.ndarray):
    '''
    Robust Scaling features
    '''
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def scale_target(y_train : np.ndarray):
    '''
    Robust Scaling target + asing transformation
    '''
    scaler = RobustScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_train_scaled = np.arcsinh(y_train_scaled)
    return y_train_scaled,scaler

def inverse_scale_target(y_pred : np.ndarray,scaler : RobustScaler):
    '''
    Just inversing back to a normal scale
    '''
    y_pred = np.sinh(y_pred)
    y = scaler.inverse_transform(y_pred)
    return y
    

def scale_X_and_split(df_train : pd.DataFrame,df_test : pd.DataFrame):
    '''
    Separates continuous and categorical features, scales only continuous variables, 
    recombines them, and then splits data into train/test sets for X and y.
    '''
    float_cols = df_train.select_dtypes(include=['float64']).columns
    nums = [col for col in float_cols if not col.startswith('fourier')]
    cats = [col for col in df_train.columns if col not in nums]  

    X_train_num = df_train[nums]
    X_test_num  = df_test[nums]
    X_train_cat = df_train[cats + ['price_eur_mwh']]
    X_test_cat = df_test[cats + ['price_eur_mwh']]
    
    X_trainNUM, y_trainNUM = reshape_df(X_train_num)
    X_trainCAT, y_trainCAT = reshape_df(X_train_cat)
    X_testNUM, y_testNUM = reshape_df(X_test_num)
    X_testCAT, y_testCAT = reshape_df(X_test_cat)
    
    n_days_train = X_trainNUM.shape[0] 
    x_num_features_train = X_trainNUM.shape[1]//24
    x_cat_features_train = X_trainCAT.shape[1]//24
    
    n_days_test = X_testNUM.shape[0]
    x_num_features_test = X_testNUM.shape[1]//24
    x_cat_features_test = X_testCAT.shape[1]//24

    
    x_num_3d_train = X_trainNUM.reshape(n_days_train,24,x_num_features_train)
    x_cat_3d_train = X_trainCAT.reshape(n_days_train,24, x_cat_features_train)
    X_train = np.concatenate([x_num_3d_train,x_cat_3d_train],axis=2)
    X_train = X_train.reshape(n_days_train, 24*(x_num_features_train + x_cat_features_train))

 
    x_num_3d_test = X_testNUM.reshape(n_days_test,24,x_num_features_test)
    x_cat_3d_test = X_testCAT.reshape(n_days_test,24,x_cat_features_test)
    X_test = np.concatenate([x_num_3d_test,x_cat_3d_test],axis=2)
    X_test = X_test.reshape(n_days_test, 24*(x_num_features_test + x_cat_features_test))

    y_train = y_trainNUM
    y_test = y_testNUM

    X_train_scaled,X_test_scaled = scale_features(X_train,X_test)
    
    return X_train_scaled,y_train,X_test_scaled,y_test