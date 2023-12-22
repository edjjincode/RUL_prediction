import numpy as np
import pandas as pd

def data_diff(data):
    df_lag = data.groupby('UnitNumber').diff(1)
    df_lag['UnitNumber'] = data['UnitNumber']
    df_lag.dropna(inplace = True)
    df_lag = df_lag.reset_index(drop = True)
    return df_lag

def monotonicity(data):

    num_pos = data[data > 0].shape[0]
    num_neg = data[data < 0].shape[0]
    tot_n = data.shape[0] - 1

    mon_val = np.abs(num_pos - num_neg)/tot_n
    return mon_val

def get_monotonicity(df_train, df_lag, sensor_cols):

    mon = []
    for col in sensor_cols:
        mon_val = []
        for unit in df_lag.UnitNumber.unique():
            mon_val.append(monotonicity(df_lag.loc[df_lag.UnitNumber == unit, col]))
        mon.append({'feature': col, 'monotonicity_val': np.mean(mon_val)})
    return pd.DataFrame(mon)

def extract_monotonicity_col(mon_df, num):
    feats = mon_df.feature[mon_df.monotonicity_val > num]

    return feats

def process_test_feats(df_test_new):
    feats_test = []
    for col in df_test_new.columns:
        if col == "UnitNumber":
            continue
        if col == "Cycle":
            continue
        feats_test.append(col)
    return feats_test