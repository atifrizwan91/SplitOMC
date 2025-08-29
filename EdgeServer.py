

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

# from collections import OrderedDict
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# class ClientsTrainingListner(FileSystemEventHandler):
#     def __init__(self, callback):
#         self.callback = callback
        
#     def on_created(self, event):
#         if not event.is_directory:
#             print(f"New file created: {event.src_path}")
#             self.process_event(event)
        
#     def on_modified(self, event):
#         if not event.is_directory:
#             print(f"File modified: {event.src_path}")
#             self.process_event(event)
            
#     def process_event(self, event):
#         time.sleep(3)
#         """Process both creation and modification events."""
#         full_path = event.src_path
#         client_id = full_path.split('\\')[0]
#         filename = os.path.basename(full_path)
#         self.callback(filename, client_id)
    

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
            server_weights.append(c.server_model.state_dict())
        
        self.last_round_clients_avg_weights = self.clients_avg_weights
        self.last_round_server_avg_weights = self.server_avg_weights
        self.clients_avg_weights = self.average_weights(clients_weights)
        self.server_avg_weights = self.average_weights(server_weights)
        
            
    def skip_aggregation(self, r):
        # Collect client and server model weights
        clients_weights = [c.client_model.state_dict() for c in self.clients]
        server_weights = [c.server_model.state_dict() for c in self.clients]
        
        # Shuffle client weights
        num_clients = len(self.clients)
        client_shuffled_indices = list(range(num_clients))
        random.shuffle(client_shuffled_indices)
        
        # Shuffle server weights (independent permutation)
        server_shuffled_indices = list(range(num_clients))
        random.shuffle(server_shuffled_indices)
        
        # Reassign shuffled weights to clients and server models
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

    # def aggregator(self, r):
    #     aggregated_model_state_dict = torch.load(os.path.join(self.clients_list[0], os.listdir(self.clients_list[0])[0]))
    #     aggregated_weights = OrderedDict({k: torch.zeros_like(v) for k, v in aggregated_model_state_dict.items()})
    #     model_count = 0
        
    #     for model_dir in self.clients_list:
    #         model_name = 'model_'+str(r)+'.pth'
    #         model_state_dict = torch.load(os.path.join(model_dir, model_name))
    #         for k, v in model_state_dict.items():
    #                 aggregated_weights[k] += v
    #         model_count += 1
    #     for k in aggregated_weights.keys():
    #         aggregated_weights[k] = aggregated_weights[k] / model_count

    #     aggregated_model_path = self.directory + '/aggregatedmodel_' + str(r) + '.pth'
    #     torch.save(aggregated_weights, aggregated_model_path)
    #     print("** Round ", r, " Aggregation Comleted")
        
    
    
    # def activate_watchdog(self):
    #     pass
            
    # def start_edge_server(self):
    #     clients_handler = ClientsTrainingListner(self.counter)
    #     clients_observer = Observer()
    #     for client in self.clients_list:
    #         print(f"Watching directory: {client}")
    #         clients_observer.schedule(clients_handler, client, recursive=False)
    #     clients_observer.start()
    #     while True:
    #         if self.current_round == self.rounds:
    #             clients_observer.stop()
    #             break
    #         time.sleep(1)
    #     clients_observer.join()
            
# es = EdgeServer(1)
# es.start_edge_server()
        