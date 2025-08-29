# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 06:00:27 2024

@author: atifr
"""
import numpy as np
import time
import torch
import copy
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
# from sklearn.metrics import precision_score
# from DataManager import client_data_info
from Models import select_model
import torch.nn as nn

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    
    
    
class Evaluator:
    def __init__(self, train_data, test_data, client):
        self.train_data = train_data
        self.test_data = test_data
        self.client = client
        
    def get_ratio_based_data(self, oop_ratio, all_selected_classes):
        test_idxs = []
        test_labels_total = np.array(self.test_data.targets)
        set_of_classes_test = set(list(all_selected_classes))
        set_of_classes_train = set(label.item() for _, labels in self.client.train_loader for label in labels)
        for c in set_of_classes_train:
            test_index_c = np.where(c == test_labels_total)[0]
            test_idxs += test_index_c.tolist()
        set_of_remaining_classes = set_of_classes_test.difference(set_of_classes_train)
        oop_classes = list(set(self.client.scope) & set(set_of_remaining_classes))
        oor_classes = set(set_of_remaining_classes) - set(self.client.scope)
        len_oop = len(oop_classes)
        len_oor = len(oor_classes)
        if len_oop > 0:    # Add oop data ratio
            number_of_samples_per_class = int(len(test_idxs) * oop_ratio / len_oop) 
            for c in oop_classes:
                test_index_c = np.where(c == test_labels_total)[0]
                test_idxs += test_index_c.tolist()[:number_of_samples_per_class]
            number_of_oop_samples = int(number_of_samples_per_class * oop_ratio)
            oor_ratio = 0.3 * oop_ratio
            if len_oor > 0:   
                number_of_samples_per_class = int((number_of_oop_samples * oor_ratio) / len_oor) 
                for c in oor_classes:
                    test_index_c = np.where(c == test_labels_total)[0]
                    test_idxs += test_index_c.tolist()[:number_of_samples_per_class]

        test_dataset_ratio = DatasetSplit(self.test_data, test_idxs)
        return test_dataset_ratio
    
    
    def average_weights(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    
    def compute_shannon_entropy(self, output):
        x = - F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        return x.sum(dim=1)
    
    def evaluate(self, eth, oop_ratio, all_selected_classes):
        client_model = copy.deepcopy(self.client.client_model)
        
        
        
        testloader_data = self.get_ratio_based_data(oop_ratio, all_selected_classes)
        testloader = DataLoader(testloader_data, batch_size=128, shuffle=False)
        
        self.client.client_model.to(self.client.device)
        self.client.client_model.eval()
        server_models = []
        for es in self.client.server_models:
            server_models.append(copy.deepcopy(self.client.server_models[es]))
        for es in server_models:
            es.eval()
            es.to(self.client.device)
            
        test_labels_total = np.array(self.test_data.targets)
        set_of_classes_test = set(test_labels_total.tolist())
        
        main_classes = set(label.item() for _, labels in self.train_data for label in labels)
        set_of_remaining_classes = set_of_classes_test.difference(main_classes)
        ood_classes = list(set(self.client.scope) & set(set_of_remaining_classes))
        oos_classes = set(set_of_remaining_classes) - set(self.client.scope)
        
        task_exit_counts = {
            'main': {'client_correct': 0, 'client_total': 0, 'server_correct': 0, 'server_total': 0},
            'ood': {'client_correct': 0, 'client_total': 0, 'server_correct': 0, 'server_total': 0},
            'oos': {'client_correct': 0, 'client_total': 0, 'server_correct': 0, 'server_total': 0}
        }
        
        loss = 0
        correct = 0
        client_correct = 0
        server_correct = 0
        total = 0
        num_of_server_side = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.client.device), labels.to(self.client.device)
        
                # Forward pass through client model
                output, representations = self.client.client_model(images)
                entpy_vals = self.compute_shannon_entropy(output)
                _, client_predictions = torch.max(output, 1)
        
                # Identify server-side samples
                server_side_idx = torch.where(entpy_vals > eth)[0]
        
                # Forward pass through server model(s)
                server_outputs = []
                for es in server_models:
                    output_server, _ = es(representations.to(self.client.device))
                    server_outputs.append(output_server)

                server_outputs = sum(server_outputs) / len(server_outputs)
                _, server_predictions = torch.max(server_outputs, 1)
                
                output[server_side_idx] = server_outputs[server_side_idx]
                
                batch_loss = self.client.criterion(output, labels)
                loss += batch_loss.item()
        
                # Overall predictions
                _, pred_labels = torch.max(output, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                # print(f'labels {labels} , pred_labels {pred_labels}')
                # Client and server overall accuracy
                client_predictions = client_predictions.view(-1)
                client_correct += torch.sum(torch.eq(client_predictions, labels)).item()
                server_predictions = server_predictions.view(-1)
                server_correct += torch.sum(torch.eq(server_predictions, labels)).item()
                num_of_server_side += len(server_side_idx)
        
                # Task-specific accuracy
                for i, label in enumerate(labels):
                    label_item = label.item()
                    # Determine task
                    if label_item in main_classes:
                        task = 'main'
                    elif label_item in ood_classes:
                        task = 'ood'
                    elif label_item in oos_classes:
                        task = 'oos'
                    else:
                        continue  # Skip if label doesn't belong to any task
        
                    # Client-side prediction
                    if i not in server_side_idx:  # Client-side if not sent to server
                        task_exit_counts[task]['client_total'] += 1
                        if client_predictions[i] == label:
                            task_exit_counts[task]['client_correct'] += 1
                    else:  # Server-side
                        task_exit_counts[task]['server_total'] += 1
                        if server_predictions[i] == label:
                            task_exit_counts[task]['server_correct'] += 1
        
        # Compute accuracies
        print(f'correct {correct} , total {total}')
        accuracy = correct / total if total > 0 else 0
        client_accuracy = client_correct / total if total > 0 else 0
        server_accuracy = server_correct / total if total > 0 else 0
        
        # Compute per-task per-exit accuracies
        task_exit_accuracies = {}
        for task in ['main', 'ood', 'oos']:
            task_exit_accuracies[task] = {
                'client': (task_exit_counts[task]['client_correct'] / task_exit_counts[task]['client_total']
                           if task_exit_counts[task]['client_total'] > 0 else 0),
                'server': (task_exit_counts[task]['server_correct'] / task_exit_counts[task]['server_total']
                           if task_exit_counts[task]['server_total'] > 0 else 0)
            }
        num_of_client_side = total - num_of_server_side;
        
        total_main = task_exit_counts['main']['client_total'] + task_exit_counts['main']['server_total']
        total_ood = task_exit_counts['ood']['client_total'] + task_exit_counts['ood']['server_total']
        total_oos = task_exit_counts['oos']['client_total'] + task_exit_counts['oos']['server_total']
        
        client_total_main = task_exit_counts['main']['client_total'] 
        client_total_ood = task_exit_counts['ood']['client_total'] 
        client_total_oos = task_exit_counts['oos']['client_total'] 
        
        server_total_main = task_exit_counts['main']['server_total']
        server_total_ood = task_exit_counts['ood']['server_total']
        server_total_oos = task_exit_counts['oos']['server_total']
        
        sample_dist = [num_of_client_side, num_of_server_side, total_main, total_ood, total_oos, client_total_main, client_total_ood, client_total_oos, server_total_main, server_total_ood, server_total_oos]
        return accuracy, client_accuracy, server_accuracy, task_exit_accuracies, sample_dist
        
    



    
    
            

    
    
