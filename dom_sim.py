import networkx as nx
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import os

from graph_helpers import read_adjacency_matrices, find_shortest_path, visualize_domains, deactivate_node, calculate_packet_delivery_ratio, find_node_disjoint_path, add_node_disjoint_path

plt.switch_backend('TkAgg')

def run_simulation(file_path, start_node, destination_node, duration, num_attacks, decrease_range=(1, 10), test_nodes=None):
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
                nodes_to_attack = random.sample(possible_nodes, k=random.randint(5, 10)) #15, 35 or 5, 10

            initial_pdr = calculate_packet_delivery_ratio(network_graph, [initial_path])  # pre-attack
            
            # Deactivate nodes
            for node_to_attack in nodes_to_attack:
                attacked_nodes.add(node_to_attack)
                deactivate_node(network_graph, node_to_attack)
                #print(f"Attacking node: {node_to_attack}")

            # Run the algorithm to adapt to the attack
            current_paths, pdr0, pdr = adapt_to_attack(network_graph, start_node, destination_node, current_paths, attacked_nodes, decrease_range, initial_pdr)

            if not current_paths.empty():
                # Extract the path, store it, and reinsert it back into the queue
                priority, initial_path = current_paths.get()
                current_paths.put((priority, initial_path))

        # Record PDR value at each second
        pdr_values_time.append(pdr0)
        pdr_values_time.append(pdr)

        # Record PDR value after each attack based on the number of attacked nodes
        if t % attacks_per_interval == 0:
            pdr_values_attack.append((len(attacked_nodes), pdr0))

    # Final path visualization and save adjacency matrix
    print("Final Path(s):")
    for priority, path in list(current_paths.queue):
        print(path)

    # Plot PDR over time and number of attacks
    plot_pdr_over_time(pdr_values_time, 60)
    plot_pdr_vs_attacks(pdr_values_attack, 60)

    #visualize_domains(network_graph, initial_path)
    save_results()

def adapt_to_attack(graph, start_node, destination_node, current_paths, attacked_nodes, decrease_range, old_pdr):
    paths = []
    while not current_paths.empty():
        paths.append(current_paths.get()[1])    
    
    # Check how much the PDR has dropped
    attack_pdr = calculate_packet_delivery_ratio(graph, paths)
    pdr_drop = old_pdr - attack_pdr
    print("old_pdr: ", old_pdr, " new_pdr: ", attack_pdr)
    
    if pdr_drop == 0:
        # If the attacked node is not in any of the current paths, disregard it
        current_paths = PriorityQueue()
        print(f"Path(s) not affected:")
        for path in paths:
            current_paths.put((-old_pdr, path))

        return current_paths, old_pdr, old_pdr
    
    if pdr_drop <= 2.0:
        # Find the best path that does not include the recently attacked node
        best_path = find_shortest_path(graph, start_node, destination_node)
        after_pdr = calculate_packet_delivery_ratio(graph, paths)
        if best_path:
            # Replace the last path with the new best path
            if paths:
                paths[-1] = best_path
            current_paths.put((-attack_pdr, best_path))
            after_pdr = calculate_packet_delivery_ratio(graph, paths)
            print(f"New (1 node change) shortest path: {best_path} after_pdr: {after_pdr}")
    else:
        # Find a completely new node-disjoint path
        new_path = find_node_disjoint_path(graph, start_node, destination_node, paths[0])
        if new_path:
            # Replace the last path with the new node-disjoint path
            if paths:
                paths[-1] = new_path
            current_paths.put((-attack_pdr, new_path))
            after_pdr = calculate_packet_delivery_ratio(graph, paths)
            print(f"Node-disjoint path found: {new_path} after_pdr: {after_pdr}")
        else:
            # If no new paths are possible, add a path to the current paths
            current_paths = PriorityQueue()            
            paths2 = add_node_disjoint_path(paths, graph, start_node, destination_node)
            after_pdr = calculate_packet_delivery_ratio(graph, paths)
            for path in paths2:
                current_paths.put((-after_pdr, path))
                print(f"Added path: {path} after_pdr: {after_pdr}")
            print(current_paths.qsize())
    

    return current_paths, attack_pdr, after_pdr

def plot_pdr_over_time(pdr_values, num_nodes):
    # Adjust x-values to reflect 0.5-second intervals
    x_values = [i / 2 for i in range(len(pdr_values))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, pdr_values, marker='o', linestyle='-', color='b')
    plt.title('Packet Delivery Ratio (PDR) Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('PDR (%)', fontsize=14)
    plt.ylim(90, 100)
    plt.grid(True)
    filename = f'results/pdr_over_time_{num_nodes}.png'
    plt.savefig(filename)
    plt.show()

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
    plt.show()

def save_results():
    if not os.path.exists('results'):
        os.makedirs('results')

if __name__ == "__main__":
    file_path = 'Test_data/MKU_files/internetworks/adjacency_60_0_7_1_updated.txt'
    start_node = 0  
    destination_node = 20 #65, 20
    duration = 25 #35
    num_attacks = 25 #35
    #test_nodes = [[14, 39], [4, 59], [11, 49], [47], [58], [23], [25, 7], [17,43], [18], [16]]  # Example list of lists for attacks
    #run_simulation(file_path, start_node, destination_node, duration, num_attacks, test_nodes=test_nodes)
    run_simulation(file_path, start_node, destination_node, duration, num_attacks) #, test_nodes=test_nodes)
