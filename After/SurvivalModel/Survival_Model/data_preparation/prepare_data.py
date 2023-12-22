
import pandas as pd

def set_fundamental():
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11',
                        's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']

    return index_names, setting_names, sensor_names, col_names, remaining_sensors, drop_sensors

def load_data(dir_path, setting_names, col_names):
    # read data
    train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path+'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    df_test = pd.read_csv((dir_path+'RUL_FD001.txt'), sep='\s+', header=None, names=['time_cycles'])

    train = add_remaining_useful_life(train)

        # clip RUL max as 125 means values in column greater than 125 becomes 125
    train['RUL'].clip(upper=125, inplace=True)

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    return train, test, df_test

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame



