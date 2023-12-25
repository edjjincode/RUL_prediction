import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


def health_indicator_regression(df_health_indicator, sensor_cols):
    model = LinearRegression()
    X = df_health_indicator[sensor_cols]
    y = df_health_indicator.HI

    model.fit(X,y)
    model.score(X,y)

    return model

def get_health_indicator(df_health_indicator, model, sensor_cols):

    df_health_indicator["HI_final"] = df_health_indicator[sensor_cols].dot(model.coef_)

    return df_health_indicator

def get_param_in_health_indicator(window, df_health_indicator):

    params_list = []
    for i in range(1, 101):
        y = df_health_indicator.HI_final[df_health_indicator.UnitNumber == i]
        cycle = df_health_indicator.Cycle[df_health_indicator.UnitNumber == i]
        theta_2, theta_1, theta_0 = np.polyfit(cycle, y, 2)
        params_list.append({'UnitNumber':i, 'theta_0': theta_0, 'theta_1': theta_1, 'theta_2': theta_2})
    params_df = pd.DataFrame(params_list, columns = ['UnitNumber', 'theta_2', 'theta_1', 'theta_0'])

    return params_df

def add_health_indicator_test_df(df_test, sensor_cols, model):
      
    df_test_1 = df_test[["UnitNumber", "Cycle"]]
    df_test = df_test[sensor_cols]
    df_test['HI'] = df_test.dot(model.coef_)
    df_test_new = pd.concat([df_test_1, df_test], axis=1)

    return df_test_new






