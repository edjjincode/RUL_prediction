from Survival_Model.data_preparation.prepare_data import set_fundamental, load_data, add_remaining_useful_life
from Survival_Model.data_preparation.process_data import add_breakdown_start, test_add_breakdown_start, cut_data_by_cycle, prepare_cox, process_cox
from Survival_Model.model.CoxPHModel import print_CoxModel, make_hazard, exponential_model, test_cox, run_curve_fit
from Survival_Model.utils.evaluation import evaluate, mse_evaluate, calculate_rmse

def main():

    #파일 경로를 지정해준다
    dir_path = '/Users/jinchan/edjjincode/NiseLabProject/Data/'

    index_names, setting_names, sensor_names, col_names, remaining_sensors, drop_sensors = set_fundamental()

    train, test, df_test = load_data(dir_path, setting_names, col_names)

    train = add_remaining_useful_life(train)
    train = add_breakdown_start(train)
    cut_off = 200
    train_censored = cut_data_by_cycle(train, cut_off)

    train_cols, predict_cols = prepare_cox(index_names, remaining_sensors)

    penalizer = 0.1
    df_hazard = make_hazard(train_censored, train_cols, penalizer)

    popt, pcov = run_curve_fit(exponential_model, df_hazard)

    drop_labels = setting_names + drop_sensors
    test = test_add_breakdown_start(test, drop_labels)

    evaluate(df_hazard['RUL'], y_hat, 'train')

    y_pred = test_cox(train_censored, train_cols, penalizer)
    y_hat = exponential_model(y_pred, *popt)
    evaluate(df_test, y_hat)


if __name__ == "__main__":
    main()






    

    






