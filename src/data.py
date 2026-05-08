import pandas as pd
import numpy as np
import pytz


def convert_to_cet(df : pd.DataFrame):
    '''
    Converting from UTC to CET and then deleting zone info for convenience 
    '''
    df['timestamp'] = pd.to_datetime(df['timestamp'], yearfirst = True)
    local_timezone = pytz.timezone('Europe/Budapest')
    df['timestamp_CET'] = df['timestamp'].dt.tz_convert(local_timezone)
    df['timestamp_CET'] = df['timestamp_CET'].dt.tz_localize(None)
    df = df.drop('timestamp',axis=1)
    df = df.set_index('timestamp_CET')
    return df

def dst_handling(df : pd.DataFrame):
  '''
  CET has irregularities at DST transitions: 
  23-hour days (spring) produce a missing hour that was linearly interpolated; 
  25-hour days (autumn) produce a duplicate hour where the second occurrence was discarded.
  '''
  df1=df.groupby([ df.index.date]).size()
  indexes_with_23_hours = df1[df1==23]
  indexes_with_25_hours = df1[df1==25]
  for date in indexes_with_23_hours.index:
    df.loc[pd.Timestamp(date)+pd.Timedelta(hours=2)] = np.nan
  df.sort_index(inplace=True)
  df = df.interpolate(method='time')
  for date in indexes_with_25_hours.index:
    df = df[~df.index.duplicated(keep="first")]
  return df

def reshape_df(df : pd.DataFrame):
    '''
    Since we produce 24 outputs at once, we should reshape accordingly
    input: df with target (hours,features + target)
    output: X - (days, features*24); y - (days, 24)
    '''
    if df.index[0].hour != 0:
        print('Problem with the index')
    df=df.copy()
    
    target = 'price_eur_mwh'

    df_X = df.drop(target,axis=1)
    df_y = df[target]
    
    arr_X = df_X.values
    arr_y = df_y.values
    num_days = arr_X.shape[0]//24
    n_features = arr_X.shape[1]    
    
    arr_3d_X = arr_X.reshape(num_days,24,n_features)
    X = arr_3d_X.reshape(num_days,24*n_features)
 
    y = arr_y.reshape(num_days,24)
    return X, y