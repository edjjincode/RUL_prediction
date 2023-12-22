import pandas as pd

import numpy as np

from scipy import optimize
import itertools

def expdegradation(parameters, cycle):

    phi = parameters[0]
    theta = parameters[1]
    beta = parameters[2]

    ht = phi + theta * np.exp(beta * cycle)
    return ht


def residuals(parameters, data, y_observed, func):

    return func(parameters, data) - y_observed

def build_exp_param_df(pca_df, param_0):

    exp_params_df = pd.DataFrame(columns = ['UnitNumber', 'phi', 'theta', 'beta'])

    exp_param = []
    for i in range(1,101):

        ht = pca_df.pc1[pca_df.UnitNumber == i]
        cycle = pca_df.cycle[pca_df.UnitNumber == i]

        OptimizeResult = optimize.least_squares(residuals, param_0, args = (cycle, ht, expdegradation))
        phi, theta, beta = OptimizeResult.x

        exp_param.append({'UnitNumber':i, 'phi': phi, 'theta': theta, 'beta': beta})

    return pd.DataFrame(exp_param)

def get_params(exp_params_df):
    phi_vals = exp_params_df.phi
    theta_vals = exp_params_df.theta
    beta_vals = exp_params_df.beta

    return phi_vals, theta_vals, beta_vals

def get_bound(phi_vals, theta_vals, beta_vals):
    lb = 25
    ub = 75
    phi_bounds = [np.percentile(phi_vals, lb), np.percentile(phi_vals, ub)]
    theta_bounds = [np.percentile(theta_vals, lb), np.percentile(theta_vals, ub)]
    beta_bounds = [np.percentile(beta_vals, lb), np.percentile(beta_vals, ub)]

    bounds = ([phi_bounds[0], theta_bounds[0], beta_bounds[0]],
          [phi_bounds[1], theta_bounds[1], beta_bounds[1]])
    
    return bounds

def get_result_df(data, param_1, bounds, threshold, y_true):

    result_test = []
    for i in data.UnitNumber.unique():

        ht = data.pc1[data.UnitNumber == i]
        cycle = data.cycle[data.UnitNumber == i]

        OptimizeResult = optimize.least_squares(residuals, param_1, bounds=bounds,
                                                args = (cycle, ht, expdegradation))
        phi, theta, beta = OptimizeResult.x
        total_cycles = np.log((threshold - phi) / theta) / beta
        RUL = total_cycles - cycle.max()

        result_test.append({'UnitNumber':i, 'phi': phi, 'theta': theta, 'beta': beta,
                                            'Pred_RUL': RUL, 'True_RUL': y_true.RUL[y_true.UnitNumber == i].values[0]})
        
        return pd.DataFrame(result_test)
    

