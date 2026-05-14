
import pandas as pd

from src.data import convert_to_cet, dst_handling, reshape_24h
from src.feature import demand_minus_solar,get_calendar_features,add_rolling_stats,add_lags,add_fourier_features,drop_unused_features,corr_filtering,cross_intreactions,calc_ramps,cross_temp
from src.evaluation import mean_mae,rmse,save_plot_predicted,save_plot_residuals,create_result_folder,save_score,save_hyperparams
from src.model import time_series_test,tuning


def main():

    df = pd.read_csv('data.csv')
    df = convert_to_cet(df)
    df = dst_handling(df)

    df = demand_minus_solar(df)
    df = get_calendar_features(df)
    #df = cross_intreactions(df)
    df = add_rolling_stats(df, target_col='price_eur_mwh')
    df = add_lags(df, columns=['price_eur_mwh','demand_forecast_mwh','solar_forecast_mwh','net_exchange_forecast_mwh','demand_minus_solar'], lags=[24,48,168])
    df = calc_ramps(df)
    df = add_fourier_features(df,'month', period = 12)
    df = drop_unused_features(df,['day','month','hour','year'])

    days_for_test_left = 45 # days for test
    n_days_valid = 45 # days for test

    #Pre-filter feautures with high corr on training subset without validation and test
    to_drop = corr_filtering(df[:-(n_days_valid + days_for_test_left)*24], threshold = 0.95)  
  
    df = df.drop(to_drop, axis=1)

    df = reshape_24h(df)
    df = cross_temp(df)

    # Tuning, if need to find good hyperparameters
    #models, best_model_params = tuning(df,alphas = [0.0001, 0.01,0.05, 0.1, 1], l1_ratios = [0.3, 0.4, 0.5, 0.7, 0.8, 0.9],days_for_test_left = days_for_test_left,n_days_valid=n_days_valid, test_size=1) # days_for_test_left needed to separate train-val from test, which gives here test set of size of 45 days 
    #alpha,l1_ratio = best_model_params['alpha'],best_model_params['l1_ratio']
    #save_hyperparams(alpha, l1_ratio)
    #print(alpha,l1_ratio)
  
    



    y_true_test, y_all_preds, test_timestamps = time_series_test(df,alpha=0.01,l1_ratio=0.3,n_days_test=45,test_size=1) # here we are running on the test set of 45 days size

    print('Mean Absolute Error', mean_mae(y_true_test, y_all_preds))
    print('Root Mean Square Error', rmse(y_true_test, y_all_preds))

    create_result_folder()
    save_score(mean_mae(y_true_test, y_all_preds), rmse(y_true_test, y_all_preds))
    save_plot_predicted(y_true_test, y_all_preds, test_timestamps)
    save_plot_residuals(y_true_test, y_all_preds, test_timestamps)




if __name__ == "__main__":
    main()