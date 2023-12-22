from Exponential_Degradation.data_preparation.prepare_data import load_data, scale_data
from Exponential_Degradation.data_preparation.process_data import average_rolling, data_diff, monotonicity, get_monotonicity, extract_monotonicity_col
from Exponential_Degradation.model.exp_degradation import expdegradation, residuals, build_exp_param_df, get_params, get_bound, get_result_df 
from Exponential_Degradation.model.pca import pca_train_df, pca_test_df, pca_test_df, get_threshold, get_threshold_std
from Exponential_Degradation.utils.evaluation import evaluate, mean_absolute_percentage_error

def main():

    #파일 경로를 지정해준다
    dir_path = '/Users/jinchan/edjjincode/NiseLabProject/Data/'

    # 트레인 데이터, 테스트 데이터, RUL 데이터를 나눠서 넣는다
    df_train, df_test, y_true = load_data(dir_path)

    # 트레인 데이터와 테스트 데이터를 스케일링한다
    sensor_cols = [f"SensorMeasure{i}" for i in range(1, 22)]
    df_train, df_test = scale_data(df_train, df_test, sensor_cols)

    # 이동평균법을 활용하여 데이터를 전처리한다.
    window = 5
    data_mean = average_rolling(df_train, window, sensor_cols)

    # monotonicity를 사용해서 필요한 데이터를 전처리한다.
    df_lag = data_diff(data_mean)
    mon_df = get_monotonicity(df_train, df_lag, sensor_cols)
    feats = extract_monotonicity_col(mon_df)

    # PCA 값을 구한다 
    num = 3
    pca_df = pca_train_df(data_mean, feats, num)

    # pca_df 값을 통해 threshold 값을 구한다
    threshold = get_threshold(pca_df)

    # param 값을 구한다. 
    param_0 = [-1, 0.01, 0.01]
    
    # pca_df를 활용하여 exp_parmms_df 값을 구한다.
    exp_params_df = build_exp_param_df(pca_df, param_0)

    data_test_mean = average_rolling(df_test, window, feats)

    num = 3
    pca_test = pca_test_df(data_test_mean, feats, num)

    #  exp_param의 파라미터 값을 구한다.
    phi_vals, theta_vals, beta_vals = get_params(exp_params_df)
    param_1 = [phi_vals.mean(), theta_vals.mean(), beta_vals.mean()]

    # 
    bounds = get_bound(phi_vals, theta_vals, beta_vals)

    result_test_df = get_result_df(pca_test, param_1, bounds, threshold, y_true)

    evaluate(result_test_df.True_RUL, result_test_df.Pred_RUL)
    mean_absolute_percentage_error(result_test_df.True_RUL, result_test_df.Pred_RUL)

if __name__ == "__main__":
    main()




