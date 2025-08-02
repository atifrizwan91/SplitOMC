# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:48:44 2024

@author: atifr
"""
import copy
import torch

class NetworkAggregation():
    def __init__(self, edge_servers):
        self.edge_servers = edge_servers
    
    def aggregate_edge_servers(self):
        clients_weights = []
        server_weights = []
        for es_id, es in self.edge_servers.items():
            clients_weights.append(copy.deepcopy(es.clients_avg_weights))
            server_weights.append(copy.deepcopy(es.server_avg_weights))
        
        self.clients_avg_weights = self.average_weights(clients_weights)
        self.server_avg_weights = self.average_weights(server_weights)
        
        
    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    