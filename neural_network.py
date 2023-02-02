from tabular_data import load_airbnb
from torch.utils.data import Dataset, DataLoader, random_split 
import torch
import numpy as np
import pandas as pd

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.features, self.label = load_airbnb(clean_data, 'Price_Night')
        self.features = self.features.select_dtypes(include=["int64", "float64"])
    
    def __getitem__(self, index):
        features = self.features.iloc[index]
        features = torch.tensor(features)
        label = self.label.iloc[index] 
        return features, label

    def __len__(self):
        return len(self.features)

def get_random_split(dataset):
    train_set, test_set = random_split(dataset, [0.7, 0.3])
    train_set, validation_set = random_split(train_set, [0.5, 0.5])
    return train_set, validation_set, test_set
    
def get_data_loader(dataset, batch_size=16):
    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(11, 1)
    
    def forward(self, features):
        return self.linear_layer(features)
    
def train(model, data_loader, epochs):
    for epoch in range(epochs):
        for batch in data_loader:
            features, label = batch
            features = features.to(torch.float32)
            prediction = model(features)
            print(prediction)

if __name__ == '__main__':
    np.random.seed(1)
    dataset = AirbnbNightlyPriceImageDataset()
    train_loader, validation_loader, test_loader = get_data_loader(dataset)
    model = NeuralNetwork()
    train(model, train_loader, 1)
    
    