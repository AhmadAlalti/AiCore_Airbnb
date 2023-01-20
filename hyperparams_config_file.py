#This configuration file contains the dictionairies of different hyperparameters for different models

#Regression models hyperparameters
SGDRegressor_rs = {'loss': ['squared_error', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
             'penalty' : ['l2', 'l1', 'elasticnet'],
             'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
             'max_iter' : [750, 1000, 1250, 1500],
             'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']}

SGDRegressor_gs = {'loss': ['epsilon_insensitive'],
             'penalty' : ['l1'],
             'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
             'max_iter' : [750, 1000, 1250],
             'learning_rate': ['optimal', 'adaptive']}

DecisionTreeRegressor_rs = {"splitter": ["best","random"],
                        "max_depth": [1, 3, 5, 7, 9, 11, 12],
                        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "max_features": [1.0, "log2", "sqrt", None],
                        "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

DecisionTreeRegressor_gs = {"splitter": ["best","random"],
                        "max_depth": [1, 3, 5, 7],
                        "min_samples_leaf": [5, 6, 7],
                        "min_weight_fraction_leaf": [0.1, 0.2],
                        "max_features": ["sqrt", None],
                        "max_leaf_nodes": [70, 80]}

RandomForestRegressor_rs = {'bootstrap': [True, False],
                        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                        'max_features': [1.0, 'sqrt'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 5, 10],
                        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

RandomForestRegressor_gs = {'bootstrap': [True],
                        'max_depth': [10, 70, 80, 110],
                        'max_features': ['sqrt'],
                        'min_samples_leaf': [4],
                        'min_samples_split': [5, 8, 10, 12],
                        'n_estimators': [400, 1000, 1200, 1400]}

GradientBoostingRegressor_rs = {'n_estimators': [500, 1000, 2000],
                         'learning_rate': [.001, 0.01, .1],
                         'max_depth': [1, 2, 4],
                         'subsample': [.5, .75, 1],
                         'random_state': [1]}

GradientBoostingRegressor_gs = {'n_estimators': [500, 1000, 2000],
                         'learning_rate': [.001, 0.01, .1],
                         'max_depth': [1, 2, 4],
                         'subsample': [.75, 1, 1.25],
                         'random_state': [1]}

LogisticRegression_gs = [    
    {'penalty': ['l2', None], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['newton-cg', 'lbfgs', 'sag']},
    {'penalty': ['l1', 'l2', None], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['saga']},
    {'penalty': ['elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['saga'], 'l1_ratio': [0, 0.1, 0.3, 0.6, 0.8, 1]}]