
import os
import re
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import random
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# from collections import OrderedDict
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from types import SimpleNamespace
# from DataLoader import loader

from Models import select_model
from Evaluator import Evaluator

class Client:
    def __init__(self, client_id, data_loader, test_data, lamda, args):
        self.client_id = client_id
        self.device = args.device
        self.lamda = lamda
        self.train_loader, self.val_loader = data_loader[0], data_loader[1]
        self.test_data = test_data
        self.client_model, self.server_model = select_model(args.dataset)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.evaluator = Evaluator(copy.deepcopy(self.train_loader), copy.deepcopy(self.test_data), self)

    def set_eservers(self, edge_servers):
        self.edge_servers = edge_servers
        self.scope = self.set_scope()
        self.server_models = {}
        for es in edge_servers: #need ID here
            self.server_models[es.es_id] = copy.deepcopy(self.server_model)
    
    def set_scope(self):
        scope = []
        for es in self.edge_servers:
            scope.extend(es.scope)
            break 
        return list(set(scope))
    
    def save_model(self):
        torch.save(self.client_model.state_dict(), f'models/{self.client_id}_client_model_weights.pth')
        
    def train(self, r):
        self.client_model.to(self.device)
        for es in self.server_models:
            self.server_models[es].to(self.device)

        self.client_model.train()
        for es in self.server_models:
            self.server_models[es].train()
        epoch_loss = []
        client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=0.01, weight_decay=1e-4)
        server_optimizers = {}
        for es in self.server_models:
            server_optimizer = torch.optim.SGD(self.server_models[es].parameters(), lr=0.01, weight_decay=1e-4)
            server_optimizers[es] = copy.deepcopy(server_optimizer)
        # if r != 1:
        #     self.update_local_model()
        for epoch in range(5):
            # print(f'** Epoch {epoch} started **')
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                client_optimizer.zero_grad()
                log_probs, representation = self.client_model(images)
                l_c = self.criterion(log_probs, labels)
                l_multi_s = []
                for es in self.server_models:
                    server_optimizers[es].zero_grad()
                    log_probs, _ = self.server_models[es](representation)
                    l_s = self.criterion(log_probs, labels)
                    l_multi_s.append(l_s)
                    
                gamma=0.5
                l_s = sum(l_multi_s)/len(l_multi_s)
                
                loss = gamma * l_c + (1-gamma) * l_s
                loss.backward()
                
                client_optimizer.step()
                for es in self.server_models:
                    server_optimizers[es].step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print(f'Epoch {epoch} Loss {sum(batch_loss)/len(batch_loss)}')
        return epoch_loss
    
    def update_local_model(self):
        client_w = self.client_model.state_dict()
        avg_client_weights = self.multi_server_aggregation()
        for key in client_w.keys():
            client_w[key] = self.lamda * client_w[key] + (1 - self.lamda) * avg_client_weights[key]
        
        self.client_model.load_state_dict(client_w)
        for es in self.edge_servers:
            if es.es_id in self.server_models:
                self.server_models[es.es_id].load_state_dict(es.server_avg_weights)
            else:
                raise KeyError(f"ServerNotFoundError: Server model for ID {es.es_id} does not exist.")
    
    def update_local_model_without_lambda(self):
        client_w = self.client_model.state_dict()
        avg_client_weights = self.multi_server_aggregation()
        self.client_model.load_state_dict(avg_client_weights)
        for es in self.edge_servers:
            if es.es_id in self.server_models:
                self.server_models[es.es_id].load_state_dict(es.server_avg_weights)
            else:
                raise KeyError(f"ServerNotFoundError: Server model for ID {es.es_id} does not exist.")
                
    def multi_server_aggregation(self):
        if len(self.edge_servers) == 1:
            return self.edge_servers[0].clients_avg_weights
        clients_weights, server_weights = [], []
        for es in self.edge_servers:
            clients_weights.append(es.clients_avg_weights)
        aggregated_cross_server_clients = self.average_weights(clients_weights)
        return aggregated_cross_server_clients
        
    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():    
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg