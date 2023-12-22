from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_data(dir_path):
    
    # Define columns
    dependent_var = ['RUL']
    index_columns_names = ["UnitNumber", "Cycle"]
    operational_settings_columns_names = ["OpSet"+str(i) for i in range(1, 4)]
    sensor_measure_columns_names = ["SensorMeasure"+str(i) for i in range(1, 22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names

    # Load training data
    df_train = pd.read_csv(dir_path + 'train_FD001.txt', delim_whitespace=True, names=input_file_column_names)

    rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
    rul.columns = ['UnitNumber', 'max']
    df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv(dir_path + 'test_FD001.txt', delim_whitespace=True, names=input_file_column_names)

    y_true = pd.read_csv(dir_path + 'RUL_FD001.txt', delim_whitespace=True,names=["RUL"])
    y_true["UnitNumber"] = y_true.index + 1

    return df_train, df_test, y_true

def scale_data(df_train, df_test, sensor_cols):
    sc = MinMaxScaler(feature_range=(0,1))
    df_train[sensor_cols] = sc.fit_transform(df_train[sensor_cols])
    df_test[sensor_cols] = sc.transform(df_test[sensor_cols])

    return df_train, df_test