#This code was used to narrow down the grid search parameters in modelling_gs.py

import joblib
import json
import os
import hyperparams_config_file as hp
import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from modelling_gs import split_data, regression_scoring

df_listing = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
np.random.seed(1)
X, y = load_airbnb(df_listing, "Price_Night")
X = X.select_dtypes(include="number")
X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]

def tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
    model = model_type()
    params = [grid_dict]
    rs_reg = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=5, verbose=10, random_state=1)
    rs_reg.fit(data_sets[0], data_sets[1])
    best_iteration_hyperparams = rs_reg.best_params_
    RMSE, MAE, R2 = regression_scoring(data_sets[4], data_sets[5], rs_reg)
    return RMSE, MAE, R2, best_iteration_hyperparams, rs_reg

def evaluate_all_models(models_and_params, data_sets):
    for model in models_and_params.items():
        RMSE, MAE, R2, best_iteration_hyperparams, gs_reg = tune_regression_model_hyperparameters(model[0], data_sets, model[1])
        regression_metrics = {"RMSE": RMSE, "MAE": MAE, "R2": R2}
        path = f"models/regression/randomized_search/{model[0].__name__}/"
        save_model(gs_reg, best_iteration_hyperparams, regression_metrics, path )
        
def save_model(model, params, metrics, folder):
    try:
        os.mkdir(folder)
        model_filename = folder + 'model.joblib'
        hyperparams_filename = folder + 'hyperparameters.json'
        metrics_filename = folder + 'metrics.json'
        joblib.dump(model, model_filename)
        with open(hyperparams_filename, 'w') as file:
            json.dump(params, file)
        with open(metrics_filename, 'w') as file:    
            json.dump(metrics, file)    
    except FileExistsError as E:
        print(E)

models_and_params = {SGDRegressor: hp.SGDRegressor_rs, 
                     DecisionTreeRegressor: hp.DecisionTreeRegressor_rs, 
                     GradientBoostingRegressor: hp.GradientBoostingRegressor_rs,
                     RandomForestRegressor: hp.RandomForestRegressor_rs}

evaluate_all_models(models_and_params, data_sets)

