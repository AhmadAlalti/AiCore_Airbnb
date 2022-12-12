#This configuration file contains the dictionairies of different hyperparameters for different models

#Regression models hyperparameters
sgd_param = {'loss': ['squared_error', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
             'penalty' : ['l2', 'l1', 'elasticnet'],
             'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
             'max_iter' : [750, 1000, 1250, 1500],
             'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']}
