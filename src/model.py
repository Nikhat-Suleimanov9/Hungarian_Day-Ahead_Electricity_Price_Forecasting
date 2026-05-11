import numpy as np
import pandas as pd
from .evaluation import mean_mae,rmse
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
from .feature import scale_target,inverse_scale_target,scale_features,split_nums_cats,get_train_test_num_cat_arr,combine_num_cat_arr
from sklearn.base import BaseEstimator



def get_TimeSeriesSplit(n_splits : int = 45, test_size : int = 1*24):
    tss = TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
    return tss
'''
def train_model(X_train : np.ndarray,y_train : np.ndarray, alpha = 0.01, l1_ratio = 0.9):
   
    reg = MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio))  
    reg.fit(X_train,y_train)
    return reg'''

def predict(model : BaseEstimator ,X_test : np.ndarray):
    y_pred = model.predict(X_test)
    return y_pred

def train_dataset_size(df : pd.DataFrame, n_days_test : int = 45):
    '''
    We split the dataset into training+val and test sets based on time. 
    The last 45 days are reserved for testing, while the training set includes all data up to 45 days before the end of the dataset.
    df_train output is used for training and validation
    '''
    df_train = df[:-n_days_test*24]
    return df_train

def tuning(df : pd.DataFrame ,alphas : list = [0.01, 0.1, 1, 10], l1_ratios : list = [0.5, 0.7, 0.9], days_for_test_left = 45, n_days_valid=45, test_size=1*24):
    '''
    Getting the best given hyperparameters
    '''
    models = []
    for alpha in alphas:
        for l1_rat in l1_ratios:
            df_train_tune = train_dataset_size(df,n_days_test=days_for_test_left)
            y_true_test, y_all_preds, test_timestamps=time_series_test(df_train_tune,alpha=alpha, l1_ratio=l1_rat,n_days_test = n_days_valid,test_size = test_size)
            model_dict = {'alpha': alpha, 'l1_ratio' : l1_rat, 'mae': mean_mae(y_true_test,y_all_preds),'rmse': rmse(y_true_test,y_all_preds)}
            models.append(model_dict)
    best_model_params  = min(models, key = lambda x: x['mae'])        
    return models, best_model_params     



def time_series_test(df : pd.DataFrame, alpha = 0.01, l1_ratio = 0.9, n_days_test : int = 45, test_size : int =1*24):
    '''
    Walk-forward implementation using an expanding window with a 1-day test size over a 45-day evaluation period.
    '''

    tss = get_TimeSeriesSplit(n_splits=n_days_test,test_size=test_size)
    
    test_predictions = []
    test_true_values=[]
    timestamps = []
    
    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        
        df_num_train, df_cat_fourier_train = split_nums_cats(train)
        df_num_test, df_cat_fourier_test = split_nums_cats(test)
        
        X_trainNUM, X_trainCAT, y_train, X_testNUM, X_testCAT, y_test = get_train_test_num_cat_arr(df_num_train, df_cat_fourier_train,df_num_test,df_cat_fourier_test)
        
        X_train_scaled_NUM,X_test_scaled_NUM = scale_features(X_trainNUM,X_testNUM)

        
        X_train_scaled = combine_num_cat_arr(X_train_scaled_NUM,X_trainCAT)
        X_test_scaled = combine_num_cat_arr(X_test_scaled_NUM,X_testCAT)
        

        
        y_train_scaled,scaler = scale_target(y_train)

        regs,selectors = train_models_filtering(X_train_scaled,y_train_scaled,alpha = alpha, l1_ratio=l1_ratio)
            
    
        y_pred_scaled = predict_filtered(regs,selectors,X_test_scaled)
        y_pred = inverse_scale_target(y_pred_scaled,scaler)
    
     
        test_predictions.append(y_pred)
        test_true_values.append(y_test)
        timestamps.append(test.index)
    
    y_all_preds = np.concatenate(test_predictions)
    y_true_test=np.concatenate(test_true_values)
    test_timestamps = np.concatenate(timestamps)
    
    y_true_test = y_true_test.ravel()
    y_all_preds = y_all_preds.ravel()
    
    return y_true_test, y_all_preds, test_timestamps


def predict_filtered(models, selectors, X_test : np.ndarray):
    preds = []
    for h in range(24):
        X_selected=selectors[h].transform(X_test)
        preds.append(models[h].predict(X_selected))
        
    y_pred = np.column_stack(preds)
    return y_pred

def train_models_filtering(X_train : np.ndarray,y_train : np.ndarray, alpha = 0.01, l1_ratio = 0.9,threshold = 0.001):
    '''
    Train an ElasticNet models
    '''

    models = []
    selectors  = []
    for h in range(24):
        selector = SelectFromModel(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42),
            threshold = threshold
        )
        
        selector.fit(X_train, y_train[:, h])
        X_selected = selector.transform(X_train)
        
        

        model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=42
            )
        model.fit(X_selected, y_train[:, h])
        models.append(model)
        selectors.append(selector)
        

    return models,selectors