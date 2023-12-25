#pywt와  scipy라이브러리 사용하기
import pywt
from scipy import fftpack
from scipy import signal
from scipy import optimize
import itertools

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale

def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

def get_dwt(lowpassfilter,df, feats):
    dwt_list = []
    for i in range(len(feats)):
        dwt = lowpassfilter(df.iloc[:, i], 0.4)
        dwt_list.append(dwt)
    return dwt_list

def make_dwt_dataframe(df, df_train, dwt_list):
    df_dwt = pd.DataFrame(dwt_list)
    df_dwt = df_dwt.T
    df_dwt.columns = df.columns

    df_1 = df_train["UnitNumber"]
    df = pd.concat([df_1, df_dwt], axis = 1)

    df_2 = df_train[["Cycle", "RUL"]]
    df_new = pd.concat([df, df_2], axis = 1)
    df_new.dropna(inplace = True)
    return df_new

def make_test_dwt_dataframe(feats, df_test, dwt_list):
    df_dwt = pd.DataFrame(dwt_list)
    df_dwt = df_dwt.T
    df_dwt.columns = feats

    df_1 = df_test["UnitNumber"]
    df = pd.concat([df_1, df_dwt], axis = 1)

    df_2 = df_test["Cycle"]
    df_new = pd.concat([df, df_2], axis = 1)
    df_new.dropna(axis = 1, inplace = True)
    return df_new

def groupby_using_hi(df_new):
    return df_new.groupby('UnitNumber').RUL.transform(lambda x: minmax_scale(x))
