import networkx as nx
import matplotlib.pyplot as plt
import random
import time

from visualize import read_transactions, find_shortest_path, visualize_graph, save_adjacency_matrix, deactivate_node, calculate_packet_delivery_ratio, find_node_disjoint_path

def run_simulation(file_path, start_node, destination_node, duration, num_attacks, decrease_range=(1, 10), test_node=None):
    network_graph = read_transactions(file_path)
    current_path = find_shortest_path(network_graph, start_node, destination_node)
    pdr = 100.0
    pdr_values = []
    attacked_nodes = set()

    for t in range(duration):
        #print(f"Time: {t}")
        
        # Attack a specified node if test_node is set, otherwise choose randomly
        if t % (duration // num_attacks) == 0:
            if test_node:
                node_to_attack = test_node
            else:
                possible_nodes = list(set(network_graph.nodes) - {start_node, destination_node})
                node_to_attack = random.choice(possible_nodes)
                
            attacked_nodes.add(node_to_attack)
            print(f"Attacking node: {node_to_attack}")
            
            # If the attacked node is in the current path, adjust the PDR
            if node_to_attack in current_path:
                old_pdr = pdr
                pdr = calculate_packet_delivery_ratio(pdr, decrease_range)
                pdr_drop = old_pdr - pdr
                print(f"Packet Delivery Ratio after attack: {pdr:.2f}%")
                
                # Dynamic thresholds for PDR drop
                if pdr_drop < 2.0:
                    current_path = find_shortest_path(network_graph, start_node, destination_node)
                    print(f"New shortest path: {current_path}")
                else:
                    current_path = find_node_disjoint_path(network_graph, start_node, destination_node, current_path)
                    if current_path is None:
                        print("No node-disjoint path found.")
                        current_path = find_alternative_path(network_graph, start_node, destination_node, attacked_nodes)
                        if current_path:
                            print(f"Alternative path found: {current_path}")
                        else:
                            print("No alternative path found. Using multiple paths.")
                            current_path = use_multiple_paths(network_graph, start_node, destination_node, attacked_nodes)
                    else:
                        print(f"Node-disjoint path: {current_path}")
                
                # Increase PDR by a random constant
                increase_percentage = random.uniform(decrease_range[0], decrease_range[1])
                pdr = min(100.0, pdr * (1 + increase_percentage / 100))
                print(f"Packet Delivery Ratio increased to: {pdr:.2f}%")
                
            else:
                print(f"No Changes to Path")
        
        # Record PDR value at each second
        pdr_values.append(pdr)
                
        # Simulate some time delay
        # time.sleep(1)

    # Plot PDR over time
    plot_pdr_over_time(pdr_values)

    # Final path visualization and save adjacency matrix
    visualize_graph(network_graph, current_path)
    save_adjacency_matrix(network_graph, 'output_after_simulation.txt')


def find_alternative_path(graph, start_node, destination_node, attacked_nodes):
    subgraph = graph.copy()
    subgraph.remove_nodes_from(attacked_nodes)
    return find_shortest_path(subgraph, start_node, destination_node)


def use_multiple_paths(graph, start_node, destination_node, attacked_nodes):
    paths = []
    subgraph = graph.copy()
    subgraph.remove_nodes_from(attacked_nodes)
    
    # Find multiple paths
    while True:
        path = find_shortest_path(subgraph, start_node, destination_node)
        if path:
            paths.append(path)
            subgraph.remove_nodes_from(path)
        else:
            break
    
    # Combine paths if possible
    combined_path = []
    for path in paths:
        combined_path.extend(path)
    return combined_path if combined_path else None


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
    file_path = 'transactions_usnet_9nodes.txt'
    start_node = 2001  
    destination_node = 2007  
    duration = 100
    num_attacks = 10
    test_node = 2003  
    run_simulation(file_path, start_node, destination_node, duration, num_attacks, test_node=test_node)
