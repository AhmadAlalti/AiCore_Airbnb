import pandas as pd
import hyperparams_config_file as hp
import numpy as np
import itertools
import glob
import joblib
import json
import os
from tabular_data import load_airbnb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1) #Is the test size correct? original 30%, split into two 15%
    return X_train, y_train, X_test, y_test, X_val, y_val

def evaluate_all_models(models_and_params, data_sets):
    for model in models_and_params.items():
        RMSE, MAE, R2, best_iteration_hyperparams, gs_reg = tune_regression_model_hyperparameters(model[0], data_sets, model[1])
        regression_metrics = {"RMSE": RMSE, "MAE": MAE, "R2": R2}
        path = f"models/regression/grid_search/{model[0].__name__}/"
        save_model(gs_reg, best_iteration_hyperparams, regression_metrics, path )

def regression_scoring(X, y, model):
    y_pred = model.predict(X)
    RMSE = mean_squared_error(y, y_pred, squared=False)
    MAE = mean_absolute_error(y, y_pred)
    R2 = r2_score(y, y_pred)
    return RMSE, MAE, R2

def tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
    model = model_type()
    params = [grid_dict]
    gs_reg = GridSearchCV(estimator=model, param_grid=params, verbose=10)
    gs_reg.fit(data_sets[0], data_sets[1])
    best_iteration_hyperparams = gs_reg.best_params_
    RMSE, MAE, R2 = regression_scoring(data_sets[4], data_sets[5], gs_reg)
    return RMSE, MAE, R2, best_iteration_hyperparams, gs_reg

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
    except FileExistsError as E: #Will this skip saving the new model if I run the training again and try to save the new model? How to get around this?
        print(E)

def find_best_model():
    metrics_files = glob.glob("./models/regression/grid_search/*/metrics.json", recursive=True)    
    best_score = 0
    for file in metrics_files:
        f = open(str(file))
        dic_metrics = json.load(f)
        score = dic_metrics['R2']
        if score > best_score:
            best_score = score
            best_name = str(file).split('/')[-2]
    path = f'./models/regression/grid_search/{best_name}/'
    model = joblib.load(path + 'model.joblib')
    with open (path + 'hyperparameters.json', 'r') as fp:
        param = json.load(fp)
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    return model, param, metrics

if __name__ == "__main__":
    df_listing = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    np.random.seed(1)
    X, y = load_airbnb(df_listing, "Price_Night")
    X = X.select_dtypes(include="number")
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]
    models_and_params = {SGDRegressor: hp.SGDRegressor_gs, 
                         DecisionTreeRegressor: hp.DecisionTreeRegressor_gs, 
                         GradientBoostingRegressor: hp.GradientBoostingRegressor_gs,
                         RandomForestRegressor: hp.RandomForestRegressor_gs}
    evaluate_all_models(models_and_params, data_sets)
    model, param, metrics = find_best_model()
    print ('best regression model is: ', model)
    print('with metrics', metrics)
    

# def custom_tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
#     keys, values = zip(*grid_dict.items())
#     iterations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     RMSE_list = []
#     for iteration in iterations_dicts:
#         loss = iteration["loss"]
#         penalty = iteration["penalty"]
#         alpha = iteration["alpha"]
#         max_iter = iteration["max_iter"]
#         learning_rate = iteration["learning_rate"]
#         model = model_type(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, learning_rate=learning_rate)
#         model.fit(data_sets[0], data_sets[1])
#         # RMSE_Train, MAE_Train, R2_Train = regression_scoring(data_sets[0], data_sets[1], model)
#         RMSE_Val, MAE_Val, R2_Val = regression_scoring(data_sets[4], data_sets[5], model)
#         RMSE_list.append(RMSE_Val)
#         if RMSE_Val <= min(RMSE_list):
#             RMSE = RMSE_Val
#             MAE = MAE_Val
#             R2 = R2_Val
#             best_iteration_hyperparams = iteration
#             best_model = model
#     return RMSE, MAE, R2, best_iteration_hyperparams, best_model