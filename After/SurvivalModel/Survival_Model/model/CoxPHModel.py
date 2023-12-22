!pip install lifelines

import matplotlib.pyplot as plt
import numpy as np
from lifelines import  CoxTimeVaryingFitter, NelsonAalenFitter, CoxPHFitter
from scipy.optimize import curve_fit

def Print_CoxModel(train_censored, train_cols,  penalizer):
    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(train_censored[train_cols], id_col="unit_nr", event_col='breakdown',
        start_col='start', stop_col='time_cycles', show_progress=True)
    ctv.print_summary()
    plt.figure(figsize=(10,5))
    ctv.plot()
    plt.show()
    plt.close()

def make_hazard(train_censored, train_cols, penalizer):
    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(train_censored[train_cols], id_col="unit_nr", event_col='breakdown',
        start_col='start', stop_col='time_cycles', show_progress=True)
    df_hazard = train_censored.copy().reset_index()
    df_hazard['hazard'] = ctv.predict_log_partial_hazard(df_hazard)
    return df_hazard

def exponential_model(z, a, b):
    return a * np.exp(-b * z)

def run_curve_fit(exponential_model, df_hazard):
    popt, pcov = curve_fit(exponential_model, df_hazard['hazard'], df_hazard['RUL'])
    return popt, pcov



    


