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
def cross_intreactions(df,columns = ['demand_forecast_mwh','solar_forecast_mwh','net_exchange_forecast_mwh']):
    df = df.copy()
    for i in range(len(columns)):
        for j in range(i+1,len(columns)):
            df[f'{columns[i]} x {columns[j]}'] = df[columns[i]] * df[columns[j]]
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
    

def split_nums_cats(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    nums = [col for col in float_cols if not col.startswith('fourier')]
    cats_fourier = [col for col in df.columns if col not in nums]
        
    df_num = df[nums]    
    df_cat_fourier =df[cats_fourier + ['price_eur_mwh']]  
    
    return df_num, df_cat_fourier

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
   
def get_train_test_num_cat_arr(df_train_num, df_train_cat,df_test_num,df_test_cat):
    X_trainNUM, y_trainNUM = reshape_df(df_train_num)
    X_trainCAT, y_trainCAT = reshape_df(df_train_cat)
    X_testNUM, y_testNUM = reshape_df(df_test_num)
    X_testCAT, y_testCAT = reshape_df(df_test_cat)
    y_train = y_trainNUM
    y_test = y_testNUM
    
    return X_trainNUM, X_trainCAT, y_train, X_testNUM, X_testCAT, y_test


def combine_num_cat_arr(X_NUM,X_CAT):
    
    n_days_train = X_NUM.shape[0] 
    x_num_features = X_NUM.shape[1]//24
    x_cat_features = X_CAT.shape[1]//24

    x_num_3d = X_NUM.reshape(n_days_train,24,x_num_features)
    x_cat_3d = X_CAT.reshape(n_days_train,24, x_cat_features)
    X = np.concatenate([x_num_3d,x_cat_3d],axis=2)
    X = X.reshape(n_days_train, 24*(x_num_features + x_cat_features))
    return X 