import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
from queue import PriorityQueue
import argparse
import numpy as np
import re

from graph_helpers import read_adjacency_matrices, find_shortest_path, visualize_domains, deactivate_node, calculate_packet_delivery_ratio, find_node_disjoint_path, add_node_disjoint_path

plt.switch_backend('TkAgg')

def run_simulation(file_path, start_node, destination_node, duration, num_attacks, th_1=98.0, decrease_range=(1, 10), test_nodes=None, min_path_len=0):
    network_graph = read_adjacency_matrices(file_path)
    pdr = 100.0
    current_paths = PriorityQueue()
    initial_path = find_shortest_path(network_graph, start_node, destination_node, min_path_len)
    print("Initial path: ", initial_path)
    current_paths.put((-pdr, initial_path))
    
    pdr_values_time = [pdr]  # For PDR over time
    pdr_values_attack = []   # For PDR as the number of nodes attacked increases
    attacked_nodes = set()
    
    # Track min and max PDR
    min_pdr = 100.0
    max_pdr = 100.0
    
    # Determine the number of attacks per time interval
    attacks_per_interval = duration // num_attacks
    
    for t in range(duration):
        #print(f"Attack #{t // attacks_per_interval + 1}")
        
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

            if new_pdr < th_1:
                current_paths, initial_pdr, new_pdr2 = adapt_to_attack(network_graph, start_node, destination_node, current_paths, th_1, new_pdr, min_path_len)

            if not current_paths.empty():
                # Extract the path, store it, and reinsert it back into the queue
                priority, initial_path = current_paths.get()
                current_paths.put((priority, initial_path))

        # Record PDR value at each second
        pdr_values_time.append(new_pdr)
        
        # Update min and max PDR values
        if new_pdr < min_pdr:
            min_pdr = new_pdr
        if new_pdr > max_pdr:
            max_pdr = new_pdr

        # Record PDR value after each attack based on the number of attacked nodes
        if t % attacks_per_interval == 0:
            pdr_values_attack.append((len(attacked_nodes), new_pdr))

    # Return the collected PDR values and other relevant data
    return pdr_values_time, min_pdr, max_pdr, th_1, len(attacked_nodes), duration

def adapt_to_attack(graph, start_node, destination_node, current_paths, th_1, new_pdr):
    paths = []
    while not current_paths.empty():
        paths.append(current_paths.get()[1])    

    if new_pdr < th_1:
        new_path = find_node_disjoint_path(graph, start_node, destination_node, paths[0], min_path_len)
        if new_path:
            paths[-1] = new_path
            current_paths.put((-new_pdr, new_path))
            new_pdr = calculate_packet_delivery_ratio(graph, paths)
            print(f"Node-disjoint path found: {new_path} new_pdr: {new_pdr}")
        else:
            current_paths = PriorityQueue()            
            paths2 = add_node_disjoint_path(paths, graph, start_node, destination_node, min_path_len)
            new_pdr = calculate_packet_delivery_ratio(graph, paths2)
            for path in paths2:
                current_paths.put((-new_pdr, path))
                print(f"Added path: {path} new_pdr: {new_pdr}")
            print(current_paths.qsize())
        

    return current_paths, new_pdr, new_pdr

def save_results_to_excel(data, file_name="simulation_results.xlsx"):
    output_dir = "results/Simulation_data"
        
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, file_name)
    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Run_{}".format(len(writer.sheets) + 1))

def plot_pdr_over_time(pdr_values, num_nodes):
    x_values = list(range(len(pdr_values)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, pdr_values, marker='o', linestyle='-', color='b')
    plt.title('Packet Delivery Ratio (PDR) Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('PDR (%)', fontsize=14)
    plt.ylim(90, 100)
    plt.grid(True)
    filename = f'results/pdr_over_time_{num_nodes}.png'
    plt.savefig(filename)

def plot_pdr_vs_attacks(pdr_values_attack, num_nodes):
    attacked_nodes, pdr_values = zip(*pdr_values_attack)
    plt.figure(figsize=(12, 6))
    plt.plot(attacked_nodes, pdr_values, marker='o', linestyle='-', color='r')
    plt.title('Packet Delivery Ratio (PDR) vs Number of Nodes Attacked', fontsize=16)
    plt.xlabel('Number of Nodes Attacked', fontsize=14)
    plt.ylabel('PDR (%)', fontsize=14)
    plt.ylim(90, 100)
    plt.grid(True)
    filename = f'results/pdr_vs_attacks_{num_nodes}.png'
    plt.savefig(filename)
    

def extract_num_nodes(file_name):
    match = re.search(r'adjacency_(\d+)_', file_name)
    if match:
        num_nodes = match.group(1)
        return int(num_nodes)
    else:
        raise ValueError("Number of nodes not found in the file name")

def run_and_record_simulations(file_path, duration, num_attacks, num_runs=10, min_path_len=0):
    file_name = os.path.basename(file_path)
    num_nodes = extract_num_nodes(file_name)
    results = []
    thresholds = [1, 2, 3, 4, 5]
    thresholds = [100 - t for t in thresholds]
    
    
    for i in range(num_runs):
        pdr_values_arr = []
        start_node, destination_node = random.sample(range(num_nodes - 1), 2)  
        for th_1 in thresholds:
            pdr_values_time, min_pdr, max_pdr, threshold_value, num_attacks_nodes, duration = run_simulation(
                file_path, start_node, destination_node, duration, num_attacks, th_1, min_path_len
            )
            pdr_values_arr.append(pdr_values_time)
            
            
            results.append({
                "File Name": file_name,
                "Start Node": start_node,
                "End Node": destination_node,
                "Min PDR": min_pdr,
                "Max PDR": max_pdr,
                "Threshold": threshold_value,
                "Number of Attacks so far": num_attacks_nodes,
                "Number of Attacks per round": num_attacks,
                "Duration": duration,
            })
        plot_3d_thresholds_pdr_time(thresholds, pdr_values_arr, duration, num_nodes, file_name, i)
    # Save the results to an Excel file
    
    
    df = pd.DataFrame(results)
    save_results_to_excel(df)
    
def plot_3d_thresholds_pdr_time(thresholds, pdr_values_time, duration, num_nodes, file_name, run_number):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range(duration + 1), thresholds)
    Z = np.array(pdr_values_time[:X.shape[0]])
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_title('3D Plot of Thresholds, PDR over Time', fontsize=16)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Threshold (th_1)', fontsize=14)
    ax.set_zlabel('PDR (%)', fontsize=14)
    filename = f'results/Simulation_data/plots/pdr_vs_thresh_vs_time_{num_nodes}_{file_name}_run_{run_number}.png'
    plt.savefig(filename)
    #plt.show()

# Main execution logic...
if __name__ == "__main__":
    num_nodes = '120'
    file_name = f'adjacency_{num_nodes}_0_7_1_updated.txt'
    file_path = f'Test_data/MKU_files/internetworks/' + file_name
    
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--th_1", type=float, default=98.0, help="Threshold value for PDR")
    parser.add_argument("--file", type=str, default=file_path, help="file definer for multiple runs")
    args = parser.parse_args()

    th_1 = args.th_1
    file_path = args.file
    
    # Extract the base file name (without directories) for plot naming
    file_name = os.path.basename(file_path).replace('.txt', '')
    
    duration = 25
    num_attacks = 25
    min_path_len = 5


    run_and_record_simulations(file_path, duration, num_attacks, min_path_len)

