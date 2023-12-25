import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale

def get_trendability_data(df_train, sensor_cols):
    trend_df_list = []
    for col in sensor_cols:
        trend_vals = []
        for unit_num in df_train['UnitNumber'].unique():
            unit_data = df_train[df_train['UnitNumber'] == unit_num]
            trend_vals.append(unit_data[['Cycle', col]].corr().iloc[0][1])
        trend_df_list.append({'feature': col, 'trendability_val': abs(sum(trend_vals) / len(trend_vals))})
    return pd.DataFrame(trend_df_list)

def extract_health_indicator_col(trend_df):
    feats = trend_df.loc[trend_df['trendability_val'] > 0.2, 'feature']

    return feats

def add_health_indicator_col(df_train, feats):
    
    df_1 = df_train[["UnitNumber"]]
    df = pd.concat([df_1, df_train[feats]], axis=1).dropna()

    df_health_indicator = pd.concat([df, df_train[["Cycle", "RUL"]]], axis=1)

    df_health_indicator['HI'] = df_health_indicator.groupby('UnitNumber')['RUL'].transform(lambda x: minmax_scale(x))

    return df_health_indicator