# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:56:30 2023

@author: Aleksandr Ostudin
"""

import pandas as pd
import os
import torch as t
from torch import nn
import math
from statistics import mean
import time
import shap
import random
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch.optim import Adam

#these all are fixed for now but this is to discuss
seed = 42
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)

PATH = os.getcwd()

""" NEURAL_GROUP """

class nnet(nn.Module):
    
    """
    This is a universal constructor for the neural net judges.
    Being called it creates a judge for SHAP analysis and RMSE evaluation
    based on randomly initiated parameters.
    """
    
    def __init__(self, size: int, in_neur: int, div_coef: int, layers_dropout: float, layers_activation: nn.Module) -> None:
        super(nnet, self).__init__()
        
        self.net_layers = []

        self.net_layers.append(nn.Linear(in_neur, math.ceil(in_neur/div_coef)))
        self.net_layers.append(nn.BatchNorm1d(math.ceil(in_neur/div_coef)))
        self.net_layers.append(self.get_activation(layers_activation[0]))  
        self.net_layers.append(nn.Dropout(layers_dropout[0]))
        self.hidd_neur = math.ceil(in_neur/div_coef)

        for i in range(size):
                
            out_neur = math.ceil(self.hidd_neur/div_coef)
            self.net_layers.append(nn.Linear(self.hidd_neur, out_neur))
            self.net_layers.append(self.get_activation(layers_activation[i]))  
            self.net_layers.append(nn.BatchNorm1d(out_neur))
            self.net_layers.append(nn.Dropout(layers_dropout[i]))
            self.hidd_neur = out_neur

        self.net_layers.append(nn.Linear(self.hidd_neur, 1))
        self.net = nn.Sequential(*self.net_layers)
        self.init_weights()
        
    def init_weights(self) -> None:
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.001)
                    
        # it is an error-bypassing solution. existing SHAP tool for working with nn
        # have a bug that overwrites the layer outcome in case of re-use of activation
        # function, causing sizes mismatch. this is torch-specific issue and to avoid it
        # we explicitly establish a separate activation function every time. 
        # more info: https://github.com/shap/shap/issues/1479  
               
    def get_activation(self, activation: nn.Module) -> nn.Module:
        return activation()
        
    def forward(self, x: t.Tensor) -> t.Tensor:
            
        out = self.net(x)
            
        return out
               
class ds(Dataset):
    
    """
    This is a standard tool to make a Torch-compatible dataset structure.
    """
    
    def __init__(self, features: t.Tensor, labels: t.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self):
        
        return len(self.features)

    def __getitem__(self, idx: int) -> t.Tensor:
        
        feature = self.features[idx]
        label = self.labels[idx]
        
        return t.Tensor(feature), t.Tensor(label)
    
    # added a separate way of exporting exclusively features to use in SHAP
    def feat_only(self) -> t.Tensor:
        return t.Tensor(self.features)
    
 
class CustomEarlyStopping:
    
    """
    This is a control tool that stops the training if it is hopelessly stuck.
    :param patience: minimal epochs to consider as stuck if no improvement observed.
    :param min_delta: minimal treshold to consider as improvement.
    """
    
    def __init__(self, patience=3, min_delta=3) -> None:

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        
        #very first value is assigned
        if self.best_loss == None:
            self.best_loss = val_loss
        
        #update the best value and reset counter if validation loss improves
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        
        #trigger the counter if it gets worse
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            
            #break the cycle if the patience is out
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True   


class judge_election:
    
    """
    This is a tool to create a set of evaluating models. At the beginning of a cycle
    the "core" set of a certain size (core_size) is established. At each iteration, 
    defined amount (rotation_rate) of models is replaced by other randomly initiated 
    models without repetition, to prevent the bias yet sustain the integrity of ensemble 
    in general. last_state is providing an info on which models from core were replaced.
    """
    
    def __init__(self, last_state: dict, num_features: int) -> None:
        
        self.last_state = last_state
        self.num_features = num_features
    
    #this is not supposed to change along the research campaign so kept static
    core_size = 5
    rotation_rate = 0.2
        
    def judge_maker(self) -> nnet:
        
        #this is kept static for now but can be considered dynamic later
        sizes = [1, 2, 3]
        div_coefs = [1.5, 2.5, 3.5]
        layers_dropouts = [0, 0.1, 0.2]
        layers_activations = [nn.LeakyReLU, nn.ELU, nn.Tanh]
        
        #the base property is selected first to adjust further layers
        selected_size = random.choice(sizes)
        
        model = nnet(size=selected_size,
                     in_neur=self.num_features,
                     div_coef=random.choice(div_coefs),
                     layers_dropout=[random.choice(
                         layers_dropouts) for i in range(selected_size)],
                     layers_activation=[random.choice(
                         layers_activations) for i in range(selected_size)])

        return model
    
    #this function makes a list of models validating that all are unique 
    def arranging(self, num_judges: int) -> list:     
        
        unique = False
        while unique == False:
        
            models = [self.judge_maker() for i in range(num_judges)]  
            unique = True
            
            for i in range(len(models)):
                
                for j in range(i+1, len(models)):
                    
                    if models[i] == models[j]:
                        
                        unique = False
                        break
                
        return models
    
    def electing(self) -> list:
        
        #initial generation of models that would be used
        
        if self.last_state == None:
                
            models = self.arranging(self.core_size)
                
            self.last_state = {'core': models,
                          'not_used_idx': []}
        
        #for all further cycles
            
        else:
            
            core = self.last_state['core']
            models = core.copy()
            rotation_batch = self.arranging(math.ceil(self.core_size*self.rotation_rate))
            to_replace = [x for x in range(self.core_size) if x not in self.last_state['not_used_idx']]
            random.shuffle(to_replace)
            to_replace = to_replace[:math.ceil(self.core_size*self.rotation_rate)]
            
            for idx, replacement in zip(to_replace, rotation_batch):
                models[idx] = replacement
                
            self.last_state = {'core': core,
                          'not_used_idx': to_replace}
            
        return models
            

class neur_trial:
    
    """
    This is a main tool to work with created models.
    :explaining: is a function that returns the averaged SHAP values for each model
    :judging: returns the average of averaged 5-fold RMSE cross-evaluation for each model
    """
    
    def __init__(self, models_list: list, ds: ds) -> None:
        
        self.time = time.time()
        self.ds = ds
        self.models_list = models_list
        self.explaining_states = None  
        self.judging_states = None  
        
        self.criterion = nn.MSELoss()
        self.num_epochs = 2500
        
    def training(self, model: nnet, trainloader: object) -> nnet:
        
        early_stopping = CustomEarlyStopping(patience=3, min_delta=3)
        optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
        
        for epoch in range(self.num_epochs):
                
            for i, data in enumerate(trainloader, 0):
        
                X, y = data
                optimizer.zero_grad()
                model.train()
                preds = model(X)
                loss_train = t.sqrt(self.criterion(preds, y.unsqueeze(1)))
                loss_train = (loss_train).mean()
                loss_train.backward()
                optimizer.step()
                    
            if epoch % 100 == 0:

                early_stopping(loss_train)
                if early_stopping.early_stop:
                    break
                
            #apart from early stop there can be a case when it stably learns above
            #threshold but it is just too bad progress, for now we skip this models
            #to accelerate the development process 
            
            if epoch == 1:
                ref = loss_train
            elif epoch == math.ceil(self.num_epochs/3):
                check = loss_train
                if ref/check < 1.15:
                    print('Early stopping: too slow')
                    break

        return model
        
    def explaining(self) -> np.array:
        
        trainloader = DataLoader(self.ds, batch_size=len(self.ds)//10)
        self.models_list = [self.training(model, trainloader) for model in self.models_list]
        self.explaining_states = [model.state_dict() for model in self.models_list]
        
        return np.mean([shap.DeepExplainer(model, self.ds.feat_only()).shap_values(self.ds.feat_only()) for model in self.models_list], axis=0)

    def judging(self) -> float:
        
        kfold = KFold(n_splits=5, shuffle=True)
        test_losses = []
            
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.ds)):
        
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
        
            trainloader = DataLoader(self.ds, batch_size=len(self.ds)//10, sampler=train_subsampler)
            testloader = DataLoader(self.ds, batch_size=len(self.ds)//10, sampler=test_subsampler)
                
            self.models_list = [self.training(model, trainloader) for model in self.models_list]
            self.judging_states = [model.state_dict() for model in self.models_list]
            
            for model in self.models_list:
                
                model_losses = []
                          
                for i, data in enumerate(testloader, 0):
                            
                    with t.no_grad():
                            
                        model.eval()
                        X, y = data
                        preds = model(X)
                        loss_test = t.sqrt(self.criterion(preds, y.unsqueeze(1)))
                        loss_test = (loss_test).mean().item()
                                
                        model_losses.append(loss_test)
                        
                test_losses.append(mean(model_losses))
                
        return mean(test_losses)

#temporal check that everything works: data preparation
         
a = pd.read_csv(fr'{PATH}\agricultural_yield_train.csv')
a = a.head(100)
c = t.Tensor(a['Yield_kg_per_hectare'])
a = a.drop('Yield_kg_per_hectare', axis=1)                        
b = t.Tensor(a.to_numpy())
d = ds(b, c)

#temporal check that everything works: launching things manually, not in a cycle

e = judge_election(None, 6)
f = e.electing()
g = neur_trial(f, d)
h = g.explaining()
i = g.judging()
j = judge_election(e.last_state, 6)
k = j.electing()
g = neur_trial(k, d)