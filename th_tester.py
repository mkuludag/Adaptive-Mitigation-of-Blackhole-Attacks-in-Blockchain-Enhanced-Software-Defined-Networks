import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from queue import PriorityQueue
import random

from graph_helpers import read_adjacency_matrices, find_shortest_path, deactivate_node, calculate_packet_delivery_ratio, find_node_disjoint_path, add_node_disjoint_path

def run_simulation(file_path, start_node, destination_node, duration, num_attacks, th_1=98.0, decrease_range=(1, 10), test_nodes=None):
    network_graph = read_adjacency_matrices(file_path)
    pdr = 100.0
    current_paths = PriorityQueue()
    initial_path = find_shortest_path(network_graph, start_node, destination_node)
    print("Initial path: ", initial_path)
    current_paths.put((-pdr, initial_path))
    
    pdr_values_time = [pdr]  # For PDR over time
    pdr_values_attack = []   # For PDR as the number of nodes attacked increases
    attacked_nodes = set()
    
    # Determine the number of attacks per time interval
    attacks_per_interval = duration // num_attacks

    for t in range(duration):
        print(f"Attack #{t // attacks_per_interval + 1}")

        # Attack a specified set of nodes if test_nodes is set, otherwise choose randomly
        if t % attacks_per_interval == 0:
            if test_nodes and t // attacks_per_interval < len(test_nodes):
                nodes_to_attack = test_nodes[t // attacks_per_interval]
            else:
                possible_nodes = list(set(network_graph.nodes) - {start_node, destination_node})
                nodes_to_attack = random.sample(possible_nodes, k=random.randint(5, 15))
            
            initial_pdr = calculate_packet_delivery_ratio(network_graph, [initial_path])
            
            # Deactivate nodes
            for node_to_attack in nodes_to_attack:
                attacked_nodes.add(node_to_attack)
                deactivate_node(network_graph, node_to_attack)

            # Recalculate PDR after the attack
            new_pdr = calculate_packet_delivery_ratio(network_graph, [initial_path])
            new_pdr = initial_pdr - (initial_pdr - new_pdr)
            new_pdr2 = None
            if new_pdr < th_1: # this is wrongggggggggggggggggggg!!!!!!!!!!!
                current_paths, initial_pdr, new_pdr2 = adapt_to_attack(network_graph, start_node, destination_node, current_paths, th_1, new_pdr)

            if not current_paths.empty():
                # Extract the path, store it, and reinsert it back into the queue
                priority, initial_path = current_paths.get()
                current_paths.put((priority, initial_path))

        # Record PDR value at each second
        pdr_values_time.append(new_pdr)
        # if new_pdr2:
        #     pdr_values_time.append(new_pdr2)

        # Record PDR value after each attack based on the number of attacked nodes
        if t % attacks_per_interval == 0:
            pdr_values_attack.append((len(attacked_nodes), new_pdr))

    return pdr_values_time

def adapt_to_attack(graph, start_node, destination_node, current_paths, th_1, new_pdr):
    paths = []
    while not current_paths.empty():
        paths.append(current_paths.get()[1])    
    
    while new_pdr < th_1:
        new_path = find_node_disjoint_path(graph, start_node, destination_node, paths[0])
        if new_path:
            paths[-1] = new_path
            current_paths.put((-new_pdr, new_path))
            new_pdr = calculate_packet_delivery_ratio(graph, paths)
            print(f"Node-disjoint path found: {new_path} new_pdr: {new_pdr}")
        else:
            current_paths = PriorityQueue()            
            paths2 = add_node_disjoint_path(paths, graph, start_node, destination_node)
            new_pdr = calculate_packet_delivery_ratio(graph, paths2)
            for path in paths2:
                current_paths.put((-new_pdr, path))
                print(f"Added path: {path} new_pdr: {new_pdr}")
            print(current_paths.qsize())
        
        if new_pdr >= th_1:
            break

    return current_paths, new_pdr, new_pdr

def plot_3d_thresholds_pdr_time(thresholds, pdr_values_time, duration):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range(duration + 1), thresholds)
    Z = np.array(pdr_values_time)

    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_title('3D Plot of Thresholds, PDR over Time', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Threshold (th_1)', fontsize=14)
    ax.set_zlabel('PDR (%)', fontsize=14)
    
    plt.show()

def main():
    file_path = 'Test_data/MKU_files/internetworks/adjacency_120_0_7_1_updated.txt'
    start_node = 0
    destination_node = 65
    duration = 35
    num_attacks = 35
    thresholds = [1, 2, 3, 4, 5]

    pdr_values_time = []

    for th_1 in thresholds:
        pdr_values = run_simulation(file_path, start_node, destination_node, duration, num_attacks, th_1)
        pdr_values_time.append(pdr_values)

    plot_3d_thresholds_pdr_time(thresholds, pdr_values_time, duration)

if __name__ == "__main__":
    main()
