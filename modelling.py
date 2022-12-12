import pandas as pd
import numpy as np
import itertools
from tabular_data import load_airbnb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1) #Is the test size correct? original 30%, split into two 15%
    return X_train, y_train, X_test, y_test, X_val, y_val

def regression_scoring(X, y, model):
    y_pred = model.predict(X)
    RMSE = mean_squared_error(y, y_pred, squared=False)
    MAE = mean_absolute_error(y, y_pred)
    R2 = r2_score(y, y_pred)
    return RMSE, MAE, R2

sgd_param = {'loss': ['squared_error', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
             'penalty' : ['l2', 'l1', 'elasticnet'],
             'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
             'max_iter' : [750, 1000, 1250, 1500],
             'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']}

def custom_tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
    keys, values = zip(*grid_dict.items())
    iterations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    RMSE_list = []
    for iteration in iterations_dicts:
        loss = iteration["loss"]
        penalty = iteration["penalty"]
        alpha = iteration["alpha"]
        max_iter = iteration["max_iter"]
        learning_rate = iteration["learning_rate"]
        model = model_type(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, learning_rate=learning_rate)
        model.fit(data_sets[0], data_sets[1])
        # RMSE_Train, MAE_Train, R2_Train = regression_scoring(data_sets[0], data_sets[1], model)
        RMSE_Val, MAE_Val, R2_Val = regression_scoring(data_sets[4], data_sets[5], model)
        RMSE_list.append(RMSE_Val)
        if RMSE_Val <= min(RMSE_list):
            RMSE = RMSE_Val
            MAE = MAE_Val
            R2 = R2_Val
            best_iteration_hyperparams = iteration
            best_model = model
    return RMSE, MAE, R2, best_iteration_hyperparams, best_model

def tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
    model = model_type()
    pass
    

if __name__ == "__main__":
    df_listing = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    np.random.seed(1)
    X, y = load_airbnb(df_listing, "Price_Night")
    X = X.select_dtypes(include="number")
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]
    reg = SGDRegressor
    RMSE, MAE, R2, best_iterations_hyperparams, best_model = custom_tune_regression_model_hyperparameters(reg, data_sets, sgd_param)
    print(f"Best model: {best_model} \nBest Hyperparameters: {best_iterations_hyperparams} \nRMSE = {RMSE} \nMAE = {MAE} \nR2 = {R2}")
    
    
    