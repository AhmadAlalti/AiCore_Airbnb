import glob
import joblib
import json
import os
import hyperparams_config_file as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabular_data import load_airbnb
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def split_data(X, y):
   
    '''It splits the data into training, testing, and validation sets
    
    Parameters
    ----------
    X
        The data to split
    y
        The target variable.
    
    Returns
    -------
        X_train, y_train, X_test, y_test, X_val, y_val
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    
    return X_train, y_train, X_test, y_test, X_val, y_val



def evaluate_regression_models(models_and_params, data_sets):
    
    '''For each model in the dictionary of models and parameters, tune the hyperparameters of the model
    using the data sets, and save the best model and its metrics to the models/regression/grid_search/
    directory.
    
    Parameters
    ----------
    models_and_params
        a dictionary of models and their hyperparameters
    data_sets
        a dictionary of the data sets we want to use to train and test the model.
    '''
    for model in models_and_params.items():
        
        RMSE, MAE, R2, best_iteration_hyperparams, estimator = tune_regression_model_hyperparameters(model[0], data_sets, model[1])
        regression_metrics = {"RMSE": RMSE, "MAE": MAE, "R2": R2}
        path = f"models/regression/grid_search/{model[0].__name__}/"
        save_model(estimator, best_iteration_hyperparams, regression_metrics, path )



def evaluate_classification_models(models_and_params, data_sets):
    
    '''For each model in the models_and_params dictionary, tune the hyperparameters of the model using the
    data_sets dictionary, and save the best model and its metrics to the
    models/classification/grid_search/{model_name}/ directory
    
    Parameters
    ----------
    models_and_params
        A dictionary of models and their hyperparameters.
    data_sets
        a dictionary of the data sets we want to use to train and test the model.
    '''
    
    for model in models_and_params.items():
        
        accuracy, precision, f1, recall, best_iteration_hyperparams, estimator = tune_classification_model_hyperparameters(model[0], data_sets, model[1])
        classification_metrics = {"Accuracy": accuracy, "Precision": precision, "f1": f1, "Recall": recall}
        path = f"models/classification/grid_search/{model[0].__name__}/"
        save_model(estimator, best_iteration_hyperparams, classification_metrics, path)



def regression_scoring(X, y, model):
    
    '''It takes in a model, a set of features, and a set of labels, and returns the RMSE, MAE, and R2 score
    of the model
    
    Parameters
    ----------
    X
        the dataframe of features
    y
        the target variable
    model
        the model to be used
    
    Returns
    -------
        RMSE, MAE, R2
    '''
    
    y_pred = model.predict(X)
    RMSE = mean_squared_error(y, y_pred, squared=False)
    MAE = mean_absolute_error(y, y_pred)
    R2 = r2_score(y, y_pred)
    
    return RMSE, MAE, R2



def classification_scoring(X, y, model):
    
    '''It takes in a model, a set of features, and a set of labels, and returns the accuracy, precision,
    f1, and recall of the model
    
    Parameters
    ----------
    X
        The input data, as a numpy array
    y
        true labels
    model
        the model you want to evaluate
    
    Returns
    -------
        the accuracy, precision, f1, and recall scores for the model.
    '''
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro") #Is my averaging correct?
    f1 = f1_score(y, y_pred, average="macro")
    recall = recall_score(y, y_pred, average="macro")
    
    return accuracy, precision, f1, recall



def tune_regression_model_hyperparameters(model_type, data_sets, grid_dict):
    
    '''This function takes in a regression model, a list of data sets, and a dictionary of hyperparameters.
    It then uses GridSearchCV to find the best hyperparameters for the model. It returns the RMSE, MAE,
    R2, best hyperparameters, and the best estimator
    
    Parameters
    ----------
    model_type
        the type of model you want to tune
    data_sets
        a list of the training and testing data sets
    grid_dict
        a dictionary of the parameters to be tuned
    
    Returns
    -------
        RMSE, MAE, R2, best_iteration_hyperparams, estimator
    '''
    
    model = model_type()
    params = [grid_dict]
    gs_reg = GridSearchCV(estimator=model, param_grid=params, verbose=10, refit=True)
    gs_reg.fit(data_sets[0], data_sets[1])
    estimator = gs_reg.best_estimator_
    best_iteration_hyperparams = gs_reg.best_params_
    RMSE, MAE, R2 = regression_scoring(data_sets[4], data_sets[5], gs_reg)
    
    return RMSE, MAE, R2, best_iteration_hyperparams, estimator



def tune_classification_model_hyperparameters(model_type, data_sets, grid):
    
    '''This function takes in a model type, data sets, and a grid of hyperparameters and returns the best
    hyperparameters, the best estimator, and the accuracy, precision, f1, and recall scores.
    
    Parameters
    ----------
    model_type
        the type of model you want to tune
    data_sets
        a list of the training and test data sets
    grid
        a dictionary of parameters to tune
    
    Returns
    -------
        The accuracy, precision, f1, recall, best_iteration_hyperparams, and estimator are being returned.
    '''
    
    model = model_type()
    gs_classification = GridSearchCV(estimator=model, param_grid=grid, verbose=10, error_score='raise', refit=True)
    gs_classification.fit(data_sets[0], data_sets[1])
    estimator = gs_classification.best_estimator_
    best_iteration_hyperparams = gs_classification.best_params_
    accuracy, precision, f1, recall = classification_scoring(data_sets[4], data_sets[5], gs_classification)
    
    return accuracy, precision, f1, recall, best_iteration_hyperparams, estimator



def save_model(model, params, metrics, folder):
    
    ''' The function takes in a model, a dictionary of hyperparameters, a dictionary of metrics, and a
    folder name. It creates a folder with the given name, and saves the model, hyperparameters, and
    metrics in that folder
    
    Parameters
    ----------
    model
        the model object
    params
        a dictionary of the hyperparameters used to train the model
    metrics
        a dictionary of metrics that you want to save.
    folder
        the folder where the model will be saved
    '''
    
    os.makedirs(folder, exist_ok=True)
    model_filename = folder + 'model.joblib'
    hyperparams_filename = folder + 'hyperparameters.json'
    metrics_filename = folder + 'metrics.json'
    joblib.dump(model, model_filename)
    
    with open(hyperparams_filename, 'w') as file:
        json.dump(params, file)
    
    with open(metrics_filename, 'w') as file:    
        json.dump(metrics, file)



def find_best_model(reg_or_class):
    
    '''It finds the best model (regression or classification) and returns the model, the hyperparameters
    and the metrics
    
    Parameters
    ----------
    reg_or_class
        "regression" or "classification"
    
    Returns
    -------
        The model, the hyperparameters and the metrics.
    '''
    
    if reg_or_class == "regression":
        
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
    
    elif reg_or_class == "classification":
        
        metrics_files = glob.glob("./models/classification/grid_search/*/metrics.json", recursive=True)    
        best_score = 0
        
        for file in metrics_files:
            
            f = open(str(file))
            dic_metrics = json.load(f)
            score = dic_metrics['f1']
            
            if score > best_score:
                best_score = score
                best_name = str(file).split('/')[-2]
        
        path = f'./models/classification/grid_search/{best_name}/'
        model = joblib.load(path + 'model.joblib')
    
    with open (path + 'hyperparameters.json', 'r') as fp:
        param = json.load(fp)
    
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    
    return model, param, metrics



if __name__ == "__main__":
    df_listing = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    np.random.seed(1)
    
    
#Regression:
    X, y = load_airbnb(df_listing, "Price_Night")
    X = X.select_dtypes(include=["int64", "float64"])
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]
    models_and_params = {SGDRegressor: hp.SGDRegressor_gs, 
                         DecisionTreeRegressor: hp.DecisionTreeRegressor_gs, 
                         GradientBoostingRegressor: hp.GradientBoostingRegressor_gs,
                         RandomForestRegressor: hp.RandomForestRegressor_gs}
    evaluate_regression_models(models_and_params, data_sets)
    model, param, metrics = find_best_model("regression")
    print ('best regression model is: ', model)
    print('with metrics', metrics)

#Classification
    X, y = load_airbnb(df_listing, "Category")
    X = X.select_dtypes(include=["int64", "float64"])
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    data_sets = [X_train, y_train, X_test, y_test, X_val, y_val]
    models_and_params = {LogisticRegression: hp.LogisticRegression_gs,
                         DecisionTreeClassifier: hp.DecisionTreeClassifier_gs,
                         RandomForestClassifier: hp.RandomForestClassifier_gs,
                         GradientBoostingClassifier: hp.GradientBoostingClassifier_gs}
    evaluate_classification_models(models_and_params, data_sets)
    model, param, metrics = find_best_model("classification")
    print ('best classification model is: ', model)
    print('with metrics', metrics)    

#Confusion Matrix for Logisic regression model
    # loaded_model = joblib.load('models/classification/grid_search/LogisticRegression/model.joblib')
    # result = loaded_model.fit(X_train, y_train)
    # y_pred = result.predict(X_val)
    # confusion_matrix_1 = confusion_matrix(y_val, y_pred)
    # cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_1)
    # cm_display.plot()
    # plt.show()

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