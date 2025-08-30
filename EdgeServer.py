

import os
import re
import copy
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import json
import random
import time
import math
from pathlib import Path
import torch



class EdgeServer():
    def __init__(self, es_id, big_lambda, SERVER_COMM = False):
        self.es_id = es_id
        self.rounds = 10
        self.current_round = 1
        self.SERVER_COMM = SERVER_COMM
        self.big_lambda = big_lambda
        self.clients_avg_weights = None
        self.server_avg_weights = None
        
        
    def set_clients(self, clients):
        self.clients = clients
        self.scope = self.set_scope()
        print(f'ID: {self.es_id}, Scope: {self.scope}')
    
    def set_scope(self):
        scope = []
        for c in self.clients:
            scope.extend(list(set(label.item() for _, labels in c.train_loader for label in labels)))
        return list(set(scope))
        
    def train(self):
        for c in self.clients:
            pass
    def save_model(self):
        torch.save(self.server_avg_weights, f'models/{self.es_id}_server_aggregated_weights.pth')
        
    def aggregate(self, r):
        clients_weights = []
        server_weights = []
        for c in self.clients:
            clients_weights.append(c.client_model.state_dict())
            server_weights.append(c.server_models[self.es_id].state_dict())
        
        self.last_round_clients_avg_weights = self.clients_avg_weights
        self.last_round_server_avg_weights = self.server_avg_weights
        self.clients_avg_weights = self.average_weights(clients_weights)
        self.server_avg_weights = self.average_weights(server_weights)
        
            
    def skip_aggregation(self, r):
        # Collect client and server model weights
        clients_weights = [c.client_model.state_dict() for c in self.clients]
        server_weights = [c.server_model.state_dict() for c in self.clients]
        
        num_clients = len(self.clients)
        client_shuffled_indices = list(range(num_clients))
        random.shuffle(client_shuffled_indices)
        
        server_shuffled_indices = list(range(num_clients))
        random.shuffle(server_shuffled_indices)

        for i, client in enumerate(self.clients):
            # Assign shuffled client model weights
            client.client_model.load_state_dict(clients_weights[client_shuffled_indices[i]])
            # Assign shuffled server model weights
            client.server_model.load_state_dict(server_weights[server_shuffled_indices[i]])
        
        # Update bookkeeping variables
        self.last_round_clients_avg_weights = self.clients_avg_weights
        self.last_round_server_avg_weights = self.server_avg_weights
        self.clients_avg_weights = None  # No averaging in skip round
        self.server_avg_weights = None  # No averaging in skip round
        
    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():    
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    
    def aggregate_network_model_with_client_model(self, network_aggregation):
        print('** Network Communication **')
        clients_weights = [network_aggregation.clients_avg_weights, self.clients_avg_weights]
        server_weights = [network_aggregation.server_avg_weights, self.server_avg_weights]
        self.clients_avg_weights = self.update_local_model(clients_weights)
        self.server_avg_weights = self.update_local_model(server_weights)
    
    def update_local_model(self, weights):
        avg_client_weights_network = weights[0]
        avg_client_weights_clients = weights[1]
    
        for key in avg_client_weights_clients.keys():
            avg_client_weights_clients[key] = self.big_lambda * avg_client_weights_clients[key] + (1 - self.big_lambda) * avg_client_weights_network[key]
        
        return avg_client_weights_clients

 
        