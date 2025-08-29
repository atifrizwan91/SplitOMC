# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:50:49 2024

@author: atifr
"""
import random
import json
import time
import csv
from tqdm import tqdm

from Client import Client
import numpy as np
# from DataLoader import loader
import pandas as pd
from DataManager import to_data_loader
from types import SimpleNamespace

from torch.utils.data import DataLoader, Dataset
from collections import Counter

from Client import Client
from EdgeServer import EdgeServer
from DataManager import DatasetManager, DatasetManagerDD
from CreateZones import create_clusters, plot_venn, plot_venn_clients, create_clusters_seq

def setup_clients_servers(clients_to_servers, servers_to_clients, clients_objects, server_objects):
    for i in servers_to_clients:
        c = [clients_objects[key] for key in servers_to_clients[i]]
        server_objects[i].set_clients(c)
        
    for i in clients_to_servers:
        es = [server_objects[key] for key in clients_to_servers[i]]
        clients_objects[i].set_eservers(es)


def main(args):
    clients = {}
    clients_to_servers, servers_to_clients = create_clusters(args.clients, args.edge_servers, args.overlap_percentage, args.existing_steup, args)
    if(args.NonIID==1):
        DM = DatasetManager
    else:
        DM = DatasetManagerDD
    
    dm = DM(args.dataset, servers_to_clients, clients_to_servers, args.clients, args.existing_steup, args.alpha)
    train_dataset, test_dataset, user_groups = dm.get_dataset(args.overlap_percentage)

    for c in clients_to_servers:
        client_data = to_data_loader(train_dataset, user_groups[c])   # Tuple -> 0: Train Data, 1: Val Data
        clients[c] = Client(c, client_data, test_dataset, args.lamda, args)
    edgeservers = {}
    for e in servers_to_clients:
        edgeservers[e] = EdgeServer(e, args.big_lambda)
        
    setup_clients_servers(clients_to_servers, servers_to_clients, clients, edgeservers)
    all_selected_classes = []
    for es_id, es in edgeservers.items():
        all_selected_classes.extend(es.scope)
    
    all_selected_classes = np.array(list(set(all_selected_classes)))
    
    losses = {c:[] for c in clients_to_servers}
    
    ethrange = [0.8] 
    oop_ratios = [0,20.2, 0.8]
    ethrange = []
    ood_ratios = []
    all_results = {'round': [], 
                     'accuracy': [], 
                     'client_all' : [],
                     'server_all': [],
                     'client_main': [],
                     'client_ood': [],
                     'client_oos' : [],
                     'server_main': [],
                     'server_ood': [],
                     'server_oos' : [],
                     'entropy': [],
                     'OOP_Ratio': [], 
                     'OOR_Ratio':[],
                     'num_of_client_side': [], 
                     'num_of_server_side':[], 
                     'total_main':[], 
                     'total_ood':[], 
                     'total_oos':[], 
                     'client_total_main':[], 
                     'client_total_ood':[], 
                     'client_total_oos':[], 
                     'server_total_main':[], 
                     'server_total_ood':[], 
                     'server_total_oos':[]}
    
    
    for r in tqdm(range(1, 301)):
        done = []
        for es_id, es in edgeservers.items():
            for c in es.clients:
                if c.client_id not in done:
                    c.train(r)
                    done.append(c.client_id)
            es.aggregate(r)
        
        if(r>297):
            ethrange = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.3]
            oop_ratios = [0.1 * i for i in range(11)]
        for eth in ethrange:
                for oop_r in oop_ratios:
                        done = []
                        accuracies_clients = []
                        client_main, client_ood, client_oos = [],[],[]
                        server_main, server_ood, server_oos = [],[],[]
                        client_all, server_all = [],[]
                        samples_distribution = []
                        for es_id, es in edgeservers.items():
                            for client in es.clients:
                                if client.client_id not in done:
                                    done.append(client.client_id)
                                    accuracy, client_accuracy, server_accuracy, task_exit_accuracies, sample_dist = client.evaluator.evaluate(eth, oop_r, all_selected_classes)
                                    client_main.append(task_exit_accuracies['main']['client'])
                                    client_ood.append(task_exit_accuracies['ood']['client'])
                                    client_oos.append(task_exit_accuracies['oos']['client'])
                                    
                                    server_main.append(task_exit_accuracies['main']['server'])
                                    server_ood.append(task_exit_accuracies['ood']['server'])
                                    server_oos.append(task_exit_accuracies['oos']['server'])
                                    samples_distribution.append(sample_dist)
                                    client_all.append(client_accuracy)
                                    server_all.append(server_accuracy)
                                    
                                    accuracies_clients.append(accuracy)
                                    
                        all_results['round'].append(r)
                        all_results['accuracy'].append(sum(accuracies_clients)/len(accuracies_clients))
                        samples_dist_ids = ['num_of_client_side', 'num_of_server_side', 'total_main', 'total_ood', 'total_oos', 'client_total_main', 'client_total_ood', 'client_total_oos', 'server_total_main', 'server_total_ood', 'server_total_oos']
                        samples_distribution = np.sum(np.array(samples_distribution), axis = 0)
                        # samples_distribution = sum(samples_distribution)/len(samples_distribution)
                        for i, val in zip(samples_dist_ids, samples_distribution):
                            all_results[i].append(val)
                        all_results['client_all'].append(sum(client_all)/len(client_all))
                        all_results['server_all'].append(sum(server_all)/len(server_all))
                        all_results['client_main'].append(sum(client_main)/len(client_main))
                        all_results['client_ood'].append(sum(client_ood)/len(client_ood))
                        all_results['client_oos'].append(sum(client_oos)/len(client_oos))
                        all_results['server_main'].append(sum(server_main)/len(server_main))
                        all_results['server_ood'].append(sum(server_ood)/len(server_ood))
                        all_results['server_oos'].append(sum(server_oos)/len(server_oos))
                        
                        all_results['entropy'].append(eth)
                        all_results['OOP_Ratio'].append(oop_r)
                        all_results['OOR_Ratio'].append(oop_r*0.3)
                        df = pd.DataFrame(all_results)
                        # print('-'*100)
                        # print('Ratio: ',str(ood_r), ' Accuracy: ', sum(accuracies_clients)/len(accuracies_clients))
        done = []
        for es_id, es in edgeservers.items():
                for c in es.clients:
                    if c.client_id not in done:
                        c.update_local_model()
                        done.append(c.client_id)
            
    
    
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'Alpha: {args.alpha} Lambda: {args.lamda} Overlapping: {args.overlap_percentage}_Dataset: {args.dataset} Resylts.csv')


with open('config.json') as json_file:
    conf = json.load(json_file)
args = SimpleNamespace(**conf)

