import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

def similarity_score_df(df_new, df_test_new, params_df):
    list_test_fit = []
    for i in df_test_new.UnitNumber.unique():
        HI = df_test_new.HI[df_test_new.UnitNumber == i]
        cycle = df_test_new.Cycle[df_test_new.UnitNumber == i]
        for j in params_df.UnitNumber.unique():
            theta_0 = params_df.theta_0[params_df.UnitNumber == j].values
            theta_1 = params_df.theta_1[params_df.UnitNumber == j].values
            theta_2 = params_df.theta_2[params_df.UnitNumber == j].values
            pred_HI = theta_0 + theta_1*cycle + theta_2*cycle*cycle
            Residual = np.mean(np.abs(pred_HI - HI))
            total_life = df_new.Cycle[df_new.UnitNumber == j].max()
            similarity_score = np.exp(-Residual*Residual)
            list_test_fit.append({'UnitNumber':i, 'Model': j, 'Residual': Residual,
                                'similarity': similarity_score, 'total_life': total_life})
    df_test_new_fit = pd.DataFrame(list_test_fit, columns=['UnitNumber', 'Model', 'Residual', 'similarity', 'total_life'])

    return df_test_new_fit

def get_top_similarity_score(num, df_test_new_fit):

    ind = df_test_new_fit.groupby('UnitNumber')['similarity'].nlargest(num).reset_index()['level_1']
    result_df = df_test_new_fit.iloc[ind]

    return result_df

def predicted_RUL(y_true, result_df, df_test_new):

    y_true_5 = y_true.copy()

    y_true_5["Pred_RUL"] = (result_df.groupby('UnitNumber')['total_life'].mean() - df_test_new.groupby('UnitNumber')['Cycle'].max()).values 

    return y_true_5