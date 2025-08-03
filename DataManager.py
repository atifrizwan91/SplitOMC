# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 01:16:03 2024

@author: atifr
"""
import random
import copy
import json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import matplotlib.pyplot as plt

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class DatasetManagerDD:
    def __init__(self, data_name, server_to_clients, clients_to_server, num_clients, preload, alpha):
        self.data_name = data_name
        self.preload = preload
        self.server_to_clients = server_to_clients
        self.clients_to_server = clients_to_server
        self.num_clients = num_clients
        self.alpha = alpha  # Dirichlet concentration parameter
        self.load_data()
        if(data_name == 'cifar100'):
            self.classes = 100
            self.filter = 0.001 
            self.es_filter = 0.001
        else:
            self.classes = 10
            self.filter = 0.1 
            self.es_filter = 0.05
    def load_data(self):
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if self.data_name == 'mnist':
            data_dir = '../data/mnist'
            self.train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)  
        if self.data_name == 'cifar10':
            data_dir = '../data/cifar10'
            self.train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        if self.data_name == 'cifar100':
            data_dir = '../data/cifar100'
            self.train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=apply_transform)
    
    def zone_scope_noniid_data(self, overlap_percentage):
        zone_scope = self.set_zone_scope(self.server_to_clients)
        all_idx = [i for i in range(len(self.train_dataset))]
        labels = self.train_dataset.targets
        idxs_labels = np.vstack((all_idx, labels))
        clients_class_probs = self.assign_class_probs_to_clients(zone_scope,self.classes)
        clients_with_classes = self.invert_dict(clients_class_probs)
        
        # Distribute samples to clients using Dirichlet probabilities
        shards = {}
        for idx, label in idxs_labels.T:
            if label not in shards:
                shards[label] = []
            shards[label].append(idx)
        
        clients_with_idx = {i: [] for i in range(1, self.num_clients + 1)}
        for label in clients_with_classes:
            clients = clients_with_classes[label]
            samples = shards[label]
            client_probs = [clients_class_probs[c][label] for c in clients]
            subshards = self.split_class_samples_to_clients(samples, clients, client_probs)
            for s, c in enumerate(clients):
                clients_with_idx[c].extend(subshards[s])
        return clients_with_idx
    
    def assign_class_probs_to_clients(self, zone_scope, total_classes):
        clients_class_probs = {i: [0] * total_classes for i in range(1, self.num_clients + 1)}
        for client_id in clients_class_probs:
            client_scope = []
            zones = self.clients_to_server[client_id]
            for z in zones:
                client_scope.extend(zone_scope[z])
                break
            client_scope = list(set(client_scope))
            
            if client_scope:
                dist = np.random.dirichlet([self.alpha] * len(client_scope))
                # Filter probabilities >= 0.2 and normalize
                filtered_probs = [p if p >= self.filter else 0 for p in dist]
                total_prob = sum(filtered_probs)
                if total_prob > 0:
                    normalized_probs = [p / total_prob for p in filtered_probs]
                else:
                    # If all probs < 0.2, assign to highest prob class
                    max_idx = np.argmax(dist)
                    normalized_probs = [1.0 if i == max_idx else 0 for i in range(len(dist))]
                
                for idx, class_id in enumerate(client_scope):
                    clients_class_probs[client_id][class_id] = normalized_probs[idx]
        
        return clients_class_probs
    
    def split_class_samples_to_clients(self, samples, clients, client_probs):
        if not clients or not samples:
            return [[] for _ in clients]
        
        # Normalize probabilities
        total_prob = sum(client_probs)
        if total_prob == 0:
            return [[] for _ in clients]
        client_probs = [p / total_prob for p in client_probs]
        
        # Distribute samples based on probabilities
        shard_sizes = (np.array(client_probs) * len(samples)).astype(int)
        
        # Ensure all samples are distributed
        total_assigned = sum(shard_sizes)
        if total_assigned < len(samples):
            max_prob_idx = np.argmax(client_probs)
            shard_sizes[max_prob_idx] += len(samples) - total_assigned
        
        subshards = []
        start_index = 0
        for size in shard_sizes:
            end_index = start_index + size
            subshards.append(samples[start_index:end_index])
            start_index = end_index
        return subshards
    
    def invert_dict(self, clients_class_probs):
        inverted_dict = {}
        total_classes = len(next(iter(clients_class_probs.values())))
        for class_id in range(total_classes):
            inverted_dict[class_id] = []
            for client_id, probs in clients_class_probs.items():
                if probs[class_id] > 0:
                    inverted_dict[class_id].append(client_id)
        return inverted_dict
    
    def set_zone_scope(self, server_to_clients):
        zones = {}
        all_classes = list(range(self.classes))
        
        # Use Dirichlet distribution to assign class probabilities to zones
        for zone_id in server_to_clients:
            dist = np.random.dirichlet([self.alpha] * self.classes)
            # Select classes with non-negligible probabilities (e.g., > 0.05)
            classes_for_zone = [all_classes[i] for i, prob in enumerate(dist) if prob > self.es_filter]
            if not classes_for_zone:  # Ensure at least one class
                classes_for_zone = [all_classes[np.argmax(dist)]]
            zones[zone_id] = classes_for_zone
        
        print(zones, 'zones')
        return zones
    
    def predefined_data(self, overlap_percentage):
        def convert_keys_to_int(x):
            if isinstance(x, dict):
                return {int(k): v for k, v in x.items()}
            return x
        print('** Predefined Data Usage **')
        with open(f'class_dist Data: {self.data_name} Overlap: {overlap_percentage} Concentration: {self.alpha}.json', 'r') as json_file:
            data_loaded = json.load(json_file, object_hook=convert_keys_to_int)
        return data_loaded
    
    def get_dataset(self, overlap_percentage):
        if self.preload:
            return self.train_dataset, self.test_dataset, self.predefined_data(overlap_percentage)
        else:
            return self.train_dataset, self.test_dataset, self.zone_scope_noniid_data(overlap_percentage)
        
class DatasetManager:
    def __init__(self, data_name, server_to_clients, clients_to_server, num_clients, preload, alpha = None):
        self.data_name = data_name
        self.preload = preload
        self.server_to_clients = server_to_clients
        self.clients_to_server = clients_to_server
        self.num_clients = num_clients
        self.load_data()
        if(data_name == 'cifar100'):
            self.classes = 100
        else:
            self.classes = 10
    def load_data(self):
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if self.data_name == 'mnist':
            data_dir = '../data/mnist'
            self.train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)  
        if self.data_name == 'cifar10':
            data_dir = '../data/cifar10'
            self.train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        if self.data_name == 'cifar100':
            data_dir = '../data/cifar100'
            self.train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=apply_transform)

    def zone_scope_noniid_data(self, overlap_percentage):
        zone_scope = self.set_zone_scope(self.server_to_clients, self.classes)
        all_idx = [i for i in range(len(self.train_dataset))]
        labels = self.train_dataset.targets
        idxs_labels = np.vstack((all_idx, labels))
        clients_cloices = {i:[] for i in range(1, self.num_clients+1)}
        for i in clients_cloices:
            client_scope = []
            zones = self.clients_to_server[i]
            # for z in zones:
            #     client_scope.extend(zone_scope[z])
            client_scope.extend(zone_scope[zones[0]])
            choice = random.sample(list(set(client_scope)), int(self.classes*0.2))
            clients_cloices[i] = choice
        clients_with_classes = self.invert_dict(clients_cloices)
        shards = {}
        for idx, label in idxs_labels.T:
            if label not in shards:
                shards[label] = []
            shards[label].append(idx)
        clients_with_idx = {i:[] for i in range(1, self.num_clients+1)}
        for label in clients_with_classes:
            n_div_clients = len(clients_with_classes[label])
            subshards = self.split_class_samples_to_clients(shards[label], n_div_clients)
            for s, c in enumerate(clients_with_classes[label]):
                clients_with_idx[c].extend(subshards[s])
        
        with open('predefined_data '+str(overlap_percentage)+'.json', 'w') as json_file:
            json.dump(clients_with_idx, json_file, cls=NumpyEncoder)
        return clients_with_idx
        
    def split_class_samples_to_clients(self, samples, clients):
        shard_length = len(samples)
        base_size = shard_length // clients
        remainder = shard_length % clients
        subshards = []
        start_index = 0
    
        for i in range(clients):
            end_index = start_index + base_size + (1 if i < remainder else 0)
            subshards.append(samples[start_index:end_index])
            start_index = end_index

        return subshards
    
    def invert_dict(self, original_dict):
        inverted_dict = {}
        for key, values in original_dict.items():
            for value in values:
                if value in inverted_dict:
                    inverted_dict[value].append(key)
                else:
                    inverted_dict[value] = [key]
        return inverted_dict
        
    def set_zone_scope(self, server_to_clients, total_classes):
        zones = {}
        all_classes = list(range(0, total_classes))
        # scope_fixed = [[0,1,2,3], [3,4,5,6]]
        for zone_id in server_to_clients:
            num_classes_for_zone = random.randint(int(self.classes*0.4), int(self.classes*0.7))  # Scope of each edgeserver will be defined here (Randomly)
            classes_for_zone = random.sample(all_classes, num_classes_for_zone)
            zones[zone_id] = classes_for_zone
           
        print(zones, 'zones')
        return zones
    
    def predefined_data(self, overlap_percentage):
        def convert_keys_to_int(x):
            if isinstance(x, dict):
                return {int(k): v for k, v in x.items()}
            return x
        print('** Predefined Data Usage **')
        with open('predefined_data '+str(overlap_percentage)+'.json', 'r') as json_file:
            data_loaded = json.load(json_file, object_hook=convert_keys_to_int)
        return data_loaded
    
    def get_dataset(self, overlap_percentage):
        if(self.preload):
            return self.train_dataset, self.test_dataset, self.predefined_data(overlap_percentage)
        else:
            return self.train_dataset, self.test_dataset, self.zone_scope_noniid_data(overlap_percentage)
       



def to_data_loader(dataset, sample_idxs):
    idxs_train = sample_idxs[:int(0.8*len(sample_idxs))]
    idxs_val = sample_idxs[int(0.8*len(sample_idxs)):]

    trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size= 32, shuffle=True)
    validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
    
    return trainloader, validloader






















