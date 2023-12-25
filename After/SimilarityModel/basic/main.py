from Similarity_Model.data_preparation.prepare_data import load_data, scale_data
from Similarity_Model.data_preparation.process_data import get_trendability_data, extract_health_indicator_col, add_health_indicator_col
from Similarity_Model.health_indicator.health_indicator import health_indicator_regression, get_health_indicator, get_param_in_health_indicator, add_health_indicator_test_df
from Similarity_Model.health_indicator.similarity_score import similarity_score_df, get_top_similarity_score, predicted_RUL
from Similarity_Model.utils.evaluation import evaluate, mean_absolute_percentage_error

import pandas as pd
import sys

def main():

    #파일 경로를 지정해준다
    dir_path = '/Users/jinchan/edjjincode/NiseLabProject/Data/'

    # 트레인 데이터, 테스트 데이터, RUL 데이터를 나눠서 넣는다
    df_train, df_test, y_true = load_data(dir_path)
    
    # 트레인 데이터와 테스트 데이터를 스케일링한다
    sensor_cols = [f"SensorMeasure{i}" for i in range(1, 22)]
    df_train, df_test = scale_data(df_train, df_test, sensor_cols)

    # trendabilty를 사용해 feature selection 과정을 거친다 
    trend_df = get_trendability_data(df_train, sensor_cols)

    # health indicator를 구한다
    feats = extract_health_indicator_col(trend_df)
    df_health_indicator = add_health_indicator_col(df_train, feats)

    # 모델을 regression을 통해 구한다
    model = health_indicator_regression(df_health_indicator, feats)

    # linear regression을 기반으로 만든 모델에 health indicator 값을 구한다
    df_health_indicator = get_health_indicator(df_health_indicator, model, feats)

    # param 값을 구한다
    window = 5
    params_df = get_param_in_health_indicator(window, df_health_indicator)

    # 테스트 데이터 셋에 health indicator 값을 넣는다
    df_test_new = add_health_indicator_test_df(df_test, feats, model)

    # simialarity score 값을 구한다
    df_test_new_fit = similarity_score_df(df_health_indicator, df_test_new, params_df)

    # similarity score 값을 갖는 것 중에 num 만큼 추출한 데이터 프레임을 뽑는다
    num = 5
    result_df = get_top_similarity_score(num, df_test_new_fit)

    # result_df 값을 활용하여 y_true_num 값을 구한다
    y_true_num = predicted_RUL(y_true, result_df, df_test_new)

    evaluate(y_true_num.RUL, y_true_num.Pred_RUL)

    mean_absolute_percentage_error(y_true_num.RUL, y_true_num.Pred_RUL)

if __name__ == "__main__":
    main()









    