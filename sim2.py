import networkx as nx
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue

from graph_helpers import read_transactions, find_shortest_path, visualize_graph, save_adjacency_matrix, deactivate_node, calculate_packet_delivery_ratio, find_node_disjoint_path, add_node_disjoint_path

plt.switch_backend('TkAgg')

def run_simulation(file_path, start_node, destination_node, duration, num_attacks, decrease_range=(1, 10), test_nodes=None):
    network_graph = read_transactions(file_path)
    pdr = 100.0
    current_paths = PriorityQueue()
    initial_path = find_shortest_path(network_graph, start_node, destination_node)
    print("Initial path: ", initial_path)
    current_paths.put((-pdr, initial_path))
    pdr_values = []
    pdr_values.append(pdr)
    attacked_nodes = set()

    for t in range(duration):
        # Attack a specified node if test_nodes is set, otherwise choose randomly
        if t % (duration // num_attacks) == 0:
            if test_nodes and t // (duration // num_attacks) < len(test_nodes):
                node_to_attack = test_nodes[t // (duration // num_attacks)]
            else:
                possible_nodes = list(set(network_graph.nodes) - {start_node, destination_node})
                node_to_attack = random.choice(possible_nodes)
            
            
            initial_pdr = calculate_packet_delivery_ratio(network_graph, [initial_path]) # pre attack
            attacked_nodes.add(node_to_attack)
            deactivate_node(network_graph, node_to_attack)
            print(f"Attacking node: {node_to_attack}")
            #print(f"Attacked node {node_to_attack} safe status: {network_graph.nodes[2006]['safe']}")  # Debug print

            
            # Run the algorithm to adapt to the attack
            current_paths, pdr0, pdr = adapt_to_attack(network_graph, start_node, destination_node, current_paths, attacked_nodes, decrease_range, initial_pdr)

            if not current_paths.empty():
                # Extract the path, store it, and reinsert it back into the queue
                priority, initial_path = current_paths.get()
                current_paths.put((priority, initial_path))
        
        # Record PDR value at each second
        pdr_values.append(pdr0)
        pdr_values.append(pdr)
                
        # Simulate some time delay
        # time.sleep(1)

    # Plot PDR over time
    plot_pdr_over_time(pdr_values)

    # Final path visualization and save adjacency matrix
    print("Final Path(s):")
    for priority, path in list(current_paths.queue):
        print(path)
    visualize_graph(network_graph, initial_path)
    # save_adjacency_matrix(network_graph, 'output_after_simulation.txt')


def adapt_to_attack(graph, start_node, destination_node, current_paths, attacked_nodes, decrease_range, old_pdr):
    paths = []
    while not current_paths.empty():
        paths.append(current_paths.get()[1])    
    
    # Check how much the PDR has dropped
    #breakpoint()
    attack_pdr = calculate_packet_delivery_ratio(graph, paths)
    pdr_drop = old_pdr - attack_pdr
    print("old_pdr: ", old_pdr, " new_pdr: ", attack_pdr)
    #print(paths[0])
    
    if pdr_drop == 0: #all(node in graph.nodes and graph.nodes[node]['safe'] for path in paths for node in path):
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


def plot_pdr_over_time(pdr_values):
    plt.figure(figsize=(10, 6))
    plt.plot(pdr_values, marker='o', linestyle='-', color='b')
    plt.title('Packet Delivery Ratio (PDR) Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('PDR (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = 'Test_data/MKU_files/transactions/transaction_adjacency_60_0_7_1_updated_nodeperisp_5.txt'
    start_node = 2001  
    destination_node = 2003
    duration = 10
    num_attacks = 10
    test_nodes = [2002, 2000, 32002, 2006, 21002, 55002] 
    run_simulation(file_path, start_node, destination_node, duration, num_attacks, test_nodes=test_nodes)
    #run_simulation(file_path, start_node, destination_node, duration, num_attacks)
