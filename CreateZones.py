# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:30:24 2024

@author: atifr
"""

import random
import json
from collections import defaultdict
    
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from collections import defaultdict


def create_clusters(NUM_CLIENTS, NUM_EDGE_SERVERS):

    clients_to_edge_servers = defaultdict(list)
    min_clients_per_server = NUM_CLIENTS // NUM_EDGE_SERVERS

    for client_id in range(1, NUM_CLIENTS + 1):
        if client_id <= min_clients_per_server * NUM_EDGE_SERVERS:
            edge_server_id = ((client_id - 1) % NUM_EDGE_SERVERS) + 1
        else:
            edge_server_id = random.randint(1, NUM_EDGE_SERVERS)
        
        clients_to_edge_servers[client_id].append(edge_server_id)
        num_additional_servers = random.randint(0, NUM_EDGE_SERVERS-1)
        additional_servers = set(random.sample(range(1, NUM_EDGE_SERVERS+1), num_additional_servers))

        if edge_server_id in additional_servers:
            additional_servers.remove(edge_server_id)
        
        clients_to_edge_servers[client_id].extend(additional_servers)
            
    edge_servers_to_clients = defaultdict(list)
    for client, servers in clients_to_edge_servers.items():
        for server in servers:
            edge_servers_to_clients[server].append(client)
    edge_servers_to_clients = dict(edge_servers_to_clients)
    
    return clients_to_edge_servers, edge_servers_to_clients


def create_clusters_seq(num_clients, num_edge_servers, percentage):
    clients_to_edge_servers = defaultdict(list)
    min_clients_per_server = num_clients // num_edge_servers
    overlap_count = int(min_clients_per_server * (percentage / 100))

    for server_id in range(1, num_edge_servers + 1):
        start_index = (server_id - 1) * min_clients_per_server + 1
        if server_id == num_edge_servers:
            end_index = num_clients  
        else:
            end_index = server_id * min_clients_per_server
        
        for client_id in range(start_index, end_index + 1):
            clients_to_edge_servers[client_id].append(server_id)
        
        if server_id < num_edge_servers:  
            for client_id in range(end_index - overlap_count + 1, end_index + 1):
                clients_to_edge_servers[client_id].append(server_id + 1)
    edge_servers_to_clients = defaultdict(list)
    for client, servers in clients_to_edge_servers.items():
        for server in servers:
            edge_servers_to_clients[server].append(client)
    edge_servers_to_clients = dict(edge_servers_to_clients)
    return clients_to_edge_servers, edge_servers_to_clients


def create_clusters(num_clients, num_edge_servers, percentage, preload, args):
    if(preload):
        print('** Predefined Cluster Usage **')
        with open(f'Exp Conf Data: {args.dataset} Overlap: {args.overlap_percentage} Concentration: {args.alpha}.json', 'r') as json_file:
            exp_env = json.load(json_file)
        clients_to_edge_servers = {} 
        for i in exp_env['Clients']:
            clients_to_edge_servers[int(i)] = exp_env['Clients'][i]['Servers']
        edge_servers_to_clients = {}
        for i in exp_env['Servers']:
            edge_servers_to_clients[int(i)] = exp_env['Servers'][i]['Clients']
        return clients_to_edge_servers, edge_servers_to_clients
    
    else:
        print('The cell-configuration is missing, Please import ES-client configuration')
    
    return clients_to_edge_servers, edge_servers_to_clients


def plot_venn(edge_servers_to_clients):
    edge_server_1_clients = set(edge_servers_to_clients[1])
    edge_server_2_clients = set(edge_servers_to_clients[2])
    edge_server_3_clients = set(edge_servers_to_clients[3])
    
    
    # Populate the sets based on the clients' edge server assignments
    

    textstr = '\n'.join([
        f"1: {', '.join(map(str, edge_server_1_clients))}",
        f"2: {', '.join(map(str, edge_server_2_clients))}",
        f"3: {', '.join(map(str, edge_server_3_clients))}",
        f"1&2: {', '.join(map(str, edge_server_1_clients & edge_server_2_clients))}",
        f"2&3: {', '.join(map(str, edge_server_2_clients & edge_server_3_clients))}",
        f"1&3: {', '.join(map(str, edge_server_1_clients & edge_server_3_clients))}",
        f"1&2&3: {', '.join(map(str, edge_server_2_clients & edge_server_3_clients & edge_server_1_clients))}",
    ])
    
    # Plotting the Venn diagram
    fig = plt.figure(figsize=(10, 7))
    venn_diagram = venn3([edge_server_1_clients, edge_server_2_clients, edge_server_3_clients],
                         set_labels=('Edge Server 1', 'Edge Server 2', 'Edge Server 3'))
    plt.gca().text(0.5, -0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title("Client Distribution Across Three Edge Servers")
    plt.show()
    fig.savefig('venn.pdf', bbox_inches='tight', dpi = 600)
    

def plot_venn_clients(clients_to_edge_servers):
    edge_server_1_clients = {client for client, servers in clients_to_edge_servers.items() if 1 in servers}
    edge_server_2_clients = {client for client, servers in clients_to_edge_servers.items() if 2 in servers}
    edge_server_3_clients = {client for client, servers in clients_to_edge_servers.items() if 3 in servers}
    
    # Plot
    fig = plt.figure(figsize=(10, 7))
    venn = venn3([edge_server_1_clients, edge_server_2_clients, edge_server_3_clients], 
                 set_labels=('Edge Server 1', 'Edge Server 2', 'Edge Server 3'))
    
    for text in venn.set_labels:
        text.set_fontsize(16)
    for text in venn.subset_labels:
        if text:  # Check if the subset label exists to avoid errors
            text.set_text("\n".join(map(str, sorted([int(client_id) for client_id in text.get_text().split(', ')]))))
            text.set_fontsize(10)
    
    plt.title("Client Distribution Across Three Edge Servers")
    plt.show()
    fig.savefig('venn_clients.pdf', bbox_inches='tight', dpi = 600)


    
if __name__ == "__main__":
    clients_to_edge_servers, edge_servers_to_clients = create_clusters(50,5)
    for client, edge_servers in clients_to_edge_servers.items():
        print(f"Client {client}: Edge Servers {edge_servers}")
    
    for server, clients in sorted(edge_servers_to_clients.items()):
        print(f"Edge Server {server}: Clients {sorted(clients)}")
        
    plot_venn(clients_to_edge_servers, edge_servers_to_clients)
    
    
