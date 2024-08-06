import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pdb
import seaborn as sns
import random


def read_transactions(file_path):
    graph = nx.Graph()
    with open(file_path, 'r') as file:
        next(file)  # Skip header line
        for line in file:
            if line.strip() == "":
                continue
            parts = line.split('\t')
            prev_as = int(parts[0])
            curr_as = int(parts[1])
            next_as = int(parts[2])
            bandwidth = float(parts[3])
            delay = float(parts[4])
            hops = int(parts[5])
            
            # extract path nodes to list
            full_path_str = parts[6]
            full_path_str = full_path_str.strip()
            full_path_str = full_path_str.strip('[]')
            node_strs = full_path_str.split(',')
            full_path = [int(node_str.strip()) for node_str in node_strs]
            
            # Add edges for each hop in the full path
            for i in range(len(full_path) - 1):
                node1 = full_path[i]
                node2 = full_path[i + 1]
                graph.add_edge(node1, node2, bandwidth=bandwidth, delay=delay, hops=hops)
                
    return graph

def find_shortest_path(graph, start_node, destination_node):
    return nx.shortest_path(graph, source=start_node, target=destination_node, weight='delay')

def visualize_graph(graph, shortest_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, ax=ax)
    edge_labels = nx.get_edge_attributes(graph, 'bandwidth') # just use bandwith for now for edges
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    # Highlight the shortest path
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=shortest_path, node_color='orange')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='orange', width=2)

    plt.show()

def save_adjacency_matrix(graph, file_name):
    adj_matrix = nx.to_pandas_adjacency(graph)
    #print(adj_matrix)
    adj_matrix.to_csv(file_name, sep='\t')


def deactivate_node(graph, node_to_kill, pdr):
    # Remove the node and its edges from the graph
    if graph.has_node(node_to_kill):
        graph.remove_node(node_to_kill)
        pdr = calculate_packet_delivery_ratio(pdr)
        print(f"Packet Delivery Ratio after attack: {pdr:.2f}%")
    else:
        print(f"Node {node_to_kill} not found in the graph.")
        pdr = -1
    
    return pdr


def calculate_packet_delivery_ratio(original_pdr, decrease_range=(3, 5)):
    decrease_percentage = random.uniform(decrease_range[0], decrease_range[1])
    new_pdr = original_pdr * (1 - decrease_percentage / 100)
    return new_pdr

def find_node_disjoint_path(graph, start_node, destination_node, current_path):
    # Create a subgraph by removing all nodes in the current path except start and destination
    nodes_to_remove = set(current_path) - {start_node, destination_node}
    subgraph = graph.copy()
    subgraph.remove_nodes_from(nodes_to_remove)
    return find_shortest_path(subgraph, start_node, destination_node)



#file_path = 'transactions_nsfnet_5nodes.txt'
file_path = 'transactions_usnet_9nodes.txt'
network_graph = read_transactions(file_path)

start_node = 2001  
destination_node = 2007  

shortest_path = find_shortest_path(network_graph, start_node, destination_node)
print("Shortest path from {} to {}: {}".format(start_node, destination_node, shortest_path))

# PDR
pdr = 100.0

# Attack
modified_graph = network_graph.copy()
node_to_kill = 2003
pdr = deactivate_node(modified_graph, node_to_kill, pdr)

# if pdr < 98.0:
#     new_shortest_path = find_shortest_path(modified_graph, start_node, destination_node)
#     print("New shortest path from {} to {}: {}".format(start_node, destination_node, new_shortest_path))
#     visualize_graph(network_graph, new_shortest_path)

if pdr < 95.0:
    shortest_path = find_node_disjoint_path(modified_graph, start_node, destination_node, shortest_path)
    if shortest_path is not None:
        print("Node-disjoint path from {} to {}: {}".format(start_node, destination_node, shortest_path))
        visualize_graph(modified_graph, shortest_path)
    else:
        print("No node-disjoint path found.")
elif pdr < 98.0:
    shortest_path = find_shortest_path(modified_graph, start_node, destination_node)
    if shortest_path is not None:
        print("New shortest path from {} to {}: {}".format(start_node, destination_node, shortest_path))
        visualize_graph(modified_graph, shortest_path)
    else:
        print("No alternative path found.")
else:
    print("PDR is above 98%, no need to switch paths.")
    


# Undo Attack
modified_graph = None
shortest_path = find_shortest_path(network_graph, start_node, destination_node)
print("Shortest path from {} to {}: {}".format(start_node, destination_node, shortest_path))


save_adjacency_matrix(network_graph, 'output_after_attack.txt')


