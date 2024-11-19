import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import random
import pandas as pd
import os
from queue import PriorityQueue
import argparse
import numpy as np
import time
from collections import deque

from new_helpers import read_adjacency_matrices, print_graph_matrices, find_shortest_disjoint_path, find_shortest_disjoint_path_with_min_length

seed = 42
random.seed(seed)
np.random.seed(seed)

# Simulation parameters
packets_per_second = 10
pdr_threshold = 0.95
global_min_path_len = 1  # Minimum path length
attack_duration = 50  # Timesteps
attack_ratio_per_second = 2  # Number of nodes attacked per second
attack_percentage = 0.2  # Percentage of packets dropped during attack
max_pdr_window = 100 # max number of elements in deque which stores the packets being recieved

historical_pdr = []


# baseline
# Dijkstra
# Stocastic (safety score)
    # ML version of Stochastic safety computation
# RL
# Deep RL

def visualize_simulation(graph, source, destination, initial_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Packet Delivery Ratio (PDR)")
    ax.set_title("Network Simulation")

    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, len(historical_pdr))  # X-axis limit based on total timesteps
    ax.set_ylim(0, 1.1)  # PDR range for clarity

    attacked_nodes_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")
    disjoint_paths_text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=10, va="top")
    current_path_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=10, va="top")

    def init():
        line.set_data([], [])
        attacked_nodes_text.set_text("")
        disjoint_paths_text.set_text("")
        current_path_text.set_text("")
        return line, attacked_nodes_text, disjoint_paths_text, current_path_text

    def animate(t):
        # Update the line data up to the current time index
        line.set_data(range(t), historical_pdr[:t])

        # Update annotations with relevant information
        current_path_text.set_text(f"Initial Path: {initial_path}")
        attacked_nodes_text.set_text(f"Attacked Nodes: {', '.join(map(str, attacked_nodes))}")
        
        disjoint_paths_text.set_text(f"Disjoint Paths: {', '.join(map(str, disjoint_paths))}")

        return line, attacked_nodes_text, disjoint_paths_text, current_path_text

    print("Num of Total attacked nodes: ", len(attacked_nodes))
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(historical_pdr) + 1, interval=250, blit=True, repeat=False)
    plt.show()

def simulate_network(graph, source, destination):
    """
    Simulate the network with sliding window PDR and attack response.
    """
    # Initialize sliding window deque and PDR
    packet_deque = deque(maxlen=max_pdr_window)
    pdr = 1.0

    # Keep track of the current path, disjoint paths, attacked nodes, and used paths
    current_path = nx.shortest_path(graph, source, destination)
    if len(current_path) < min_path_len:
        print(f"Current path length ({len(current_path)}) is less than minimum ({min_path_len}). Finding a longer path.")
        current_path = find_shortest_path_with_min_length(graph, source, destination, min_path_len)
    disjoint_paths = []
    attacked_nodes = set()
    used_paths = {tuple(current_path)}  # Store initial path as used

    # Simulate for 200 timesteps
    for t in range(attack_duration):
        # Send packets
        packets_sent = packets_per_second
        packets_received = packets_per_second

        # Check for attacks and update attacked nodes
        num_attacks = attack_ratio_per_second
        for _ in range(num_attacks):
            node_to_attack = random.choice([node for node in graph.nodes() if node != source and node != destination])
            attacked_nodes.add(node_to_attack)
            #print("!!!!!!!!!node attacked: ", node_to_attack)

        # Adjust packets received based on current path and attacked nodes
        for node in current_path:
            if node in attacked_nodes:
                packets_received = int(packets_received - packets_per_second * attack_percentage)
                       
            
        # Update packet deque and calculate PDR
        packet_deque.append(packets_received / packets_sent)
        
        pdr = sum(packet_deque) / len(packet_deque)
        historical_pdr.append(pdr)
        # Print current status
        print(f"Time: {t}, PDR: {pdr:.2f}, Current Path: {current_path}")

        # Check if PDR drops below threshold
        if pdr < pdr_threshold:
            # Find a disjoint path with minimum length that hasn’t been used before
            disjoint_path = find_shortest_disjoint_path_with_min_length(
                graph, source, destination, current_path, min_path_len, used_paths
            )
            if disjoint_path:
                print(f"Switching to disjoint path: {disjoint_path}")
                current_path = disjoint_path
                disjoint_paths.append(disjoint_path)
                used_paths.add(tuple(disjoint_path))  # Mark this path as used
            else:
                print("No disjoint path found.")

        time.sleep(0.03)  # Introduce a delay for visualization purposes

    return current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr



def find_shortest_path_with_min_length(graph, source, destination, min_length):
    """
    Find the shortest path between source and destination that is at least min_length long.
    """
    paths = list(nx.all_shortest_paths(graph, source=source, target=destination))
    for path in paths:
        if len(path) >= min_length:
            return path
    return None



if __name__ == "__main__":
    file_name = "adjacency_60_0_7_1_updated.txt"
    file_path = os.path.join("Test_data", "MKU_files", "internetworks", file_name)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--N", type=int, default=60, help="Number of Nodes in Input txt file")
    parser.add_argument("--file", type=str, default=file_path, help="File path for the adjacency matrices")
    parser.add_argument("--source", type=int, default=3, help="Source node")
    parser.add_argument("--destination", type=int, default=59, help="Destination node")
    parser.add_argument("--min-path-len", type=int, default=global_min_path_len, help="Minimum path length")
    args = parser.parse_args()

    num_nodes = args.N
    file_path = args.file
    source = args.source
    destination = args.destination
    min_path_len = args.min_path_len

    # Read the adjacency matrices and create the graph
    network_graph = read_adjacency_matrices(file_path)

    # Run the simulation and visualize the results
    current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr = simulate_network(network_graph, source, destination)
    visualize_simulation(network_graph, source, destination, current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr)
