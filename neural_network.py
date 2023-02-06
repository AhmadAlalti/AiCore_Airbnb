import glob
import itertools
import json
import os
import time
import torch
import yaml
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import OrderedDict
from datetime import datetime
from tabular_data import load_airbnb
from torch.utils.data import Dataset, DataLoader, random_split 
from torchmetrics import R2Score
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2)



class AirbnbNightlyPriceImageDataset(Dataset):
    
    '''This class inherits from torch Dataset to create the appropriate dataset for the PyTorch Module 
    and Neural Network.
    '''
   
   
   
    def __init__(self):
        
        '''This function loads the data from the clean_tabular_data.csv file and then selects
        the features and labels.
        '''
        
        super().__init__()
        clean_data = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(clean_data, 'Price_Night')
        self.features = self.features.select_dtypes(include=["int64", "float64"])
    

    
    def __getitem__(self, index):
        
        '''The function takes in an index, and returns a tuple of the features and label for that index
        
        Parameters
        ----------
        index
            The index of the data point you want to get
        
        Returns
        -------
            The features and label of the data at the index.
        '''
        
        features = self.features.iloc[index]
        features = torch.tensor(features).float()
        label = self.label.iloc[index]
        label = torch.tensor(label).float()
        return features, label
    
    
    
    def __len__(self):
        
        '''`__len__` is a special function that returns the length of the object
        
        Returns
        -------
            The length of the features list.
        '''
        
        return len(self.features)



class NeuralNetwork(torch.nn.Module):
    
    '''This class represents how the neural network layers are configured and passes the features data
    into the neural network.
    '''
    
    
    def __init__(self):
       
        '''Constructor function that initializes the parent and child classes. It holds a dictionary that has the hidden layers.
        It has a for loop that creates the hidden layers and adds them to the dictionary. These layers are used to
        create a sequential object.
        '''
        
        super().__init__()
        self.in_feature = 5
        self.out_feature = 5 # edit based on config width
        self.linear_input_layer = torch.nn.Linear(11,5)
        self.linear_output_layer = torch.nn.Linear(5,1)
        self.ordered_dict = OrderedDict({'linear_input_layer':self.linear_input_layer,'ReLU':torch.nn.ReLU()})
        self.depth = 5 # edit based on config depth
        self.hidden_layer_depth = [x for x in range(self.depth)]
        
        for hidden_layer in self.hidden_layer_depth:
            self.ordered_dict['hidden_layer_'+ str(hidden_layer)] = torch.nn.Linear(self.in_feature,self.out_feature)
            self.ordered_dict['ReLU_'+ str(hidden_layer)] = torch.nn.ReLU()
            
        self.ordered_dict['linear_output_layer'] = self.linear_output_layer
        self.layers = torch.nn.Sequential(self.ordered_dict)    
    
    
    
    def forward(self, features):
        
        '''`This method passes the features data into neural network
        
        Parameters
        ----------
        features
            The input features to the network.
        
        Returns
        -------
            The output of the last layer of the network.
        '''
        
        return self.layers(features)
    
    
    
def train(model, data_loader, optimiser, epochs=15):
    
    '''It takes a model, a data loader, an optimiser, and the number of epochs to train for, and returns
    the RMSE and R2 scores for the training and validation sets, as well as the training duration and
    inference latency
    
    Parameters
    ----------
    model
        The model we want to train
    data_loader
        a dictionary containing the training and validation data loaders
    optimiser
        The optimiser to use.
    epochs, optional
        number of epochs to train for
    
    Returns
    -------
        RMSE_train, R2_train, training_duration, inference_latency, R2_val, RMSE_val
    '''
    
    writer = SummaryWriter()
    batch_idx = 0
    batch_idx2 = 0
    pred_time = []
    start_time = time.time()
    
    for epoch in range(epochs):
        
        for batch in data_loader['train']:
            features, label = batch
            label = torch.unsqueeze(label, 1)
            time_b4_pred = time.time()
            prediction = model(features)
            time_after_pred = time.time()
            time_elapsed = time_after_pred - time_b4_pred
            pred_time.append(time_elapsed)
            loss = F.mse_loss(prediction, label)
            R2_train = R2Score()
            R2_train = R2_train(prediction, label)
            RMSE_train = torch.sqrt(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalars(optimiser.__class__.__name__, {"Train_loss": loss.item()}, batch_idx)
            batch_idx += 1
            
        end_time = time.time()
        
        for batch in data_loader['validation']:
            features, label = batch
            label = torch.unsqueeze(label, 1)
            prediction = model(features)
            loss_val = F.mse_loss(prediction, label)
            writer.add_scalars(optimiser.__class__.__name__, {"Val_loss": loss_val.item()}, batch_idx2)
            R2_val = R2Score()
            R2_val = R2_val(prediction, label)
            RMSE_val = torch.sqrt(loss_val)
            batch_idx2 += 1
            
    training_duration = end_time - start_time
    inference_latency = sum(pred_time)/len(pred_time)
    
    return RMSE_train, R2_train, training_duration, inference_latency, R2_val, RMSE_val



def generate_nn_configs():
    
    '''It takes a list of optimisers, and for each optimiser, it creates a list of dictionaries, where each
    dictionary contains the parameters for a neural network
    
    Returns
    -------
        A list of dictionaries. Each dictionary contains the parameters for a single neural network.
    '''
    
    optimisers = ['Adadelta', 'SGD', 'Adam', 'Adagrad']
    all_params = []
    
    for optimiser in optimisers:
        
        if optimiser == 'Adadelta':
            params = {'optimiser': ['Adadelta'], 'learning_rate': [1.0, 0.001, 0.0001], 'rho': [0.9, 0.7, 0.3], 'weight_decay': [0, 0.5, 1, 1.5], 'width': [5], 'depth': [5]}  
       
        elif optimiser == 'SGD':
            params = {'optimiser': ['SGD'], 'learning_rate': [0.001, 0.0001], 'momentum': [0, 0.1, 0.3, 0.7], 'weight_decay': [0, 0.5, 1, 1.5], 'width': [5], 'depth': [5]}  
        
        elif optimiser == 'Adam':
            params = {'optimiser': ['Adam'], 'learning_rate': [0.001, 0.0001], 'weight_decay': [0, 0.5, 1, 1.5], 'amsgrad': [True, False], 'width': [5], 'depth': [5]}  
       
        elif optimiser == 'Adagrad':
            params = {'optimiser': ['Adagrad'], 'learning_rate': [0.01, 0.001, 0.0001], 'lr_decay': [0, 0.1, 0.3, 0.7], 'weight_decay': [0, 0.5, 1, 1.5], 'width': [5], 'depth': [5]}  
        
        keys, values = zip(*params.items())
        params_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_params.append(params_dict)    
    
    return all_params



def convert_all_params_to_yaml(all_params, yaml_file):
    
    '''It takes a dictionary of parameters and writes it to a yaml file
    
    Parameters
    ----------
    all_params
        a dictionary of all the parameters
    yaml_file
        the file to write the parameters to
    '''
    
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(all_params, f, sort_keys=False, default_flow_style=False)



def get_nn_config(yaml_file):
    
    '''The function takes in a yaml file and returns a dictionary of the yaml file's
    contents. 
    
    Parameters
    ----------
    yaml_file
        The path to the yaml file that contains the configuration for the neural network.
    
    Returns
    -------
        A dictionary of the yaml file
    '''
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
        return config 



def get_random_split(dataset):
    
    '''It takes a dataset and splits it into three sets: a training set, a validation set, and a test set
    
    Parameters
    ----------
    dataset
        The dataset to split.
    
    Returns
    -------
        a tuple of three datasets, the training set, the validation set, and the test set.
    '''
    
    train_set, test_set = random_split(dataset, [0.7, 0.3])
    train_set, validation_set = random_split(train_set, [0.5, 0.5])
    
    return train_set, validation_set, test_set
    
    
    
def get_data_loader(dataset, batch_size=32):
    
    '''It takes a dataset and returns a dictionary of data loaders for the training, validation, and test
    sets
    
    Parameters
    ----------
    dataset
        The dataset to be split.
    batch_size, optional
        The number of samples in each batch.
    
    Returns
    -------
        A dictionary with three keys: train, validation, and test. Each key has a value of a DataLoader
    object.
    '''
    
    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    data_loader = {"train": train_loader, "validation": validation_loader, "test": test_loader}
    
    return data_loader
        
        
        
def save_model(model, folder, optimiser, performance_metrics, optimiser_params):
    
    '''It takes a Pytorch model, an optimiser, a dictionary of hyperparameters and a dictionary of
    performance metrics and saves them in a folder with a timestamp
    
    Parameters
    ----------
    model
        the model you want to save
    folder
        The folder where you want to save the model
    optimiser
        the optimiser used to train the model
    performance_metrics
        a dictionary of performance metrics, e.g. {'accuracy': 0.9, 'loss': 0.1}
    optimiser_params
        a dictionary of the optimiser parameters
    '''
    
    if not isinstance(model, torch.nn.Module):
        print("Your model is not a Pytorch Module")    
    
    else:
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        optimiser_name = optimiser.__class__.__name__
        model_folder = os.path.join(folder + '/', time + '_' + optimiser_name + '/')
        os.makedirs(model_folder, exist_ok=True)
        model_path = model_folder + 'model.pt'
        sd = model.state_dict()
        torch.save(sd, model_path)
        
        with open(f"{model_folder}/hyperparameters.json", 'w') as fp:
            json.dump(optimiser_params, fp)
        
        with open(f"{model_folder}/metrics.json", 'w') as fp:
            json.dump(performance_metrics, fp)  



def get_nn_scoring(model, dataset, optimiser):
    
    '''It takes a model, a dataset, and an optimiser, and returns a dictionary of performance metrics
    
    Parameters
    ----------
    model
        the model we want to train
    dataset
        the name of the dataset to use.
    optimiser
        The optimiser used to train the model.
    
    Returns
    -------
        The performance metrics of the model.
    '''
    
    data_loader = get_data_loader(dataset)
    RMSE_train, R2_train, td_train, il_train, R2_val, RMSE_val = train(model, data_loader, optimiser)
    performance_metrics = {'RMSE_Loss_Train': RMSE_train.item(), 'R2_Score_Train': R2_train.item(), 
                           'RMSE_Loss_Validation': RMSE_val.item(), 'R2_Score_Val': R2_val.item(),
                           'training_duration_seconds': td_train, 'inference_latency_seconds': il_train}
    
    return performance_metrics



def get_optimiser(yaml_file, model):
    
    '''It takes a yaml file and a model as input, and returns a list of optimisers
    
    Parameters
    ----------
    yaml_file
        The file where the parameters are stored.
    model
        The model to train.
    
    Returns
    -------
        A list of optimisers
    '''
    
    all_params = generate_nn_configs()
    convert_all_params_to_yaml(all_params, yaml_file)
    config = get_nn_config(yaml_file)
    optimisers = []
   
    for optimiser_category in config:
        
        for optimiser in optimiser_category:
            
            if optimiser['optimiser'] == 'Adadelta':
                optim = torch.optim.Adadelta(model.parameters(), lr=optimiser['learning_rate'], rho=optimiser['rho'], weight_decay=optimiser['weight_decay'])
                optimisers.append(optim)
            
            elif optimiser['optimiser'] == 'SGD':
                optim = torch.optim.SGD(model.parameters(), lr=optimiser['learning_rate'], momentum=optimiser['momentum'], weight_decay=optimiser['weight_decay'])
                optimisers.append(optim)
            
            elif optimiser['optimiser'] == 'Adam':
                optim = torch.optim.Adam(model.parameters(), lr=optimiser['learning_rate'], weight_decay=optimiser['weight_decay'], amsgrad=optimiser['amsgrad'])
                optimisers.append(optim)
            
            elif optimiser['optimiser'] == 'Adagrad':
                optim = torch.optim.Adagrad(model.parameters(), lr=optimiser['learning_rate'], lr_decay=optimiser['lr_decay'], weight_decay=optimiser['weight_decay'])
                optimisers.append(optim)
    
    return optimisers



def find_best_nn(yaml_file, model, dataset, folder):
    
    '''It takes a model, a dataset, and a yaml file, and returns the name of the best model, the
    hyperparameters of the best model, and the performance metrics of the best model
    
    Parameters
    ----------
    yaml_file
        the path to the yaml file containing the hyperparameters
    model
        the model you want to train
    dataset
        the dataset you want to use
    folder
        the folder where the model will be saved
    
    Returns
    -------
        The name of the best model, the hyperparameters of the best model and the metrics of the best
    model.
    '''
    
    optimisers = get_optimiser(yaml_file, model)
    
    for optimiser in optimisers:
        performance_metrics = get_nn_scoring(model, dataset, optimiser)
        sd = optimiser.state_dict()
        optimiser_params = sd['param_groups'][0]
        save_model(model, folder, optimiser, performance_metrics, optimiser_params)
    
    metrics_files = glob.glob("./neural_networks/regression/*/metrics.json", recursive=True)
    best_score = 0
    
    for file in metrics_files:
        f = open(str(file))
        dic_metrics = json.load(f)
        f.close()
        score = dic_metrics['R2_Score_Val']
        if score > best_score:
            best_score = score
            best_name = str(file).split('/')[-2]
    path = f'./neural_networks/regression/{best_name}/'
    
    with open (path + 'hyperparameters.json', 'r') as fp:
        params = json.load(fp)
    
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    
    return best_name, params, metrics    
    
    
    
def print_model_info(best_name, metrics, params):
    
    '''It prints the name of the best model, the metrics, and the parameters
    
    Parameters
    ----------
    best_name
        the name of the best model that the dataset was trained on.
    metrics
        a dictionary of the metrics we want to print out
    params
        the parameters of the model.
    '''
    
    print('-' * 80)
    print(f'The best regression model was trained on: {best_name}')
    print('-' * 80)
    print('Metrics')
    for key, value in metrics.items():
        print(key, ' : ', value)
    print('-' * 80)
    print('Parameters')
    for key, value in params.items():
        if key == 'params':
            pass
        else:
            print(key, ' : ', value)    
    print('-' * 80)
        
    
    
if __name__ == '__main__':
    np.random.seed(1)
    dataset = AirbnbNightlyPriceImageDataset()
    model = NeuralNetwork()
    best_name, params, metrics = find_best_nn('nn_config.yaml', model, dataset,'neural_networks/regression')
    print_model_info(best_name, metrics, params)



        
    

    
    