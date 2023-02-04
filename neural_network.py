from tabular_data import load_airbnb
from torch.utils.data import Dataset, DataLoader, random_split 
from torchmetrics import R2Score
from datetime import datetime
from collections import OrderedDict
import torch.nn.functional as F
import itertools
import torch
import yaml
import time
import json
import glob
import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(2)

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(clean_data, 'Price_Night')
        self.features = self.features.select_dtypes(include=["int64", "float64"])
    
    def __getitem__(self, index):
        features = self.features.iloc[index]
        features = torch.tensor(features).float()
        label = self.label.iloc[index]
        label = torch.tensor(label).float()
        return features, label

    def __len__(self):
        return len(self.features)

def get_random_split(dataset):
    train_set, test_set = random_split(dataset, [0.7, 0.3])
    train_set, validation_set = random_split(train_set, [0.5, 0.5])
    return train_set, validation_set, test_set
    
def get_data_loader(dataset, batch_size=32):
    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    data_loader = {"train": train_loader, "validation": validation_loader, "test": test_loader}
    return data_loader

def get_nn_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        return config   

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
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
        return self.layers(features)
    
def train(model, data_loader, optimiser, epochs=15):
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
        
def save_model(model, folder, optimiser, performance_metrics, optimiser_params):
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
    data_loader = get_data_loader(dataset)
    RMSE_train, R2_train, td_train, il_train, R2_val, RMSE_val = train(model, data_loader, optimiser)
    performance_metrics = {'RMSE_Loss_Train': RMSE_train.item(), 'R2_Score_Train': R2_train.item(), 
                           'RMSE_Loss_Validation': RMSE_val.item(), 'R2_Score_Val': R2_val.item(),
                           'training_duration_seconds': td_train, 'inference_latency_seconds': il_train}
    return performance_metrics

def get_optimiser(yaml_file, model):
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

def generate_nn_configs():
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
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(all_params, f, sort_keys=False, default_flow_style=False)

def find_best_nn(yaml_file, model, dataset, folder):
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


    
if __name__ == '__main__':
    np.random.seed(1)
    dataset = AirbnbNightlyPriceImageDataset()
    model = NeuralNetwork()
    best_name, params, metrics = find_best_nn('nn_config.yaml', model, dataset,'neural_networks/regression')
    print ('best regression model was trained on: ', best_name)
    print('with metrics', metrics)
    print('with params', params)

        
    

    
    