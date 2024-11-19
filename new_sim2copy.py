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

from functools import reduce

def product_of_deque(deque):
    return reduce(lambda x, y: x * y, deque, 1)

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# Simulation parameters
packets_per_second = 10
pdr_threshold = 0.95
global_min_path_len = 1
attack_duration = 50
attack_ratio_per_second = 4
attack_percentage = 0.2
max_pdr_window = 25

# Historical PDR storage
historical_pdr = []
historical_pdr_no_intervention = []  # For baseline

def simulate_no_intervention(graph, source, destination, attack_duration, attack_ratio_per_second):
    """
    Simulate the network without intervention to establish a baseline.
    """
    packet_deque = deque(maxlen=max_pdr_window)
    pdr_no_intervention = 1.0
    attacked_nodes = set()

    # Find initial path
    current_path = nx.shortest_path(graph, source, destination)

    # Simulate for the same duration as the adaptive simulation
    for t in range(attack_duration):
        packets_sent = packets_per_second
        packets_received = packets_per_second

        # Randomly attack nodes each second
        for _ in range(attack_ratio_per_second):
            node_to_attack = random.choice([node for node in graph.nodes() if node != source and node != destination])
            attacked_nodes.add(node_to_attack)

        # Calculate packets received based on attacked nodes in the fixed path
        print("attacked nodes: ", attacked_nodes)
        print(f"Attack Details: packets_received = {packets_received}, packets_per_second = {packets_per_second}, attack_percentage = {attack_percentage}")
        for node in current_path:
            if node in attacked_nodes:
                packets_received = int(packets_received - packets_per_second * attack_percentage)
        print("packets recieved: ", packets_received)
        # Update deque and calculate PDR for baseline
        packet_deque.append(packets_received / packets_sent)
        product = product_of_deque(packet_deque)
        pdr_no_intervention = product #/ len(packet_deque) # sum(packet_deque) / len(packet_deque)
        print("Deque state at timestep", t, ": ", list(packet_deque))
        historical_pdr_no_intervention.append(pdr_no_intervention)

    return [1.0] + historical_pdr_no_intervention

def visualize_simulation(graph, source, destination, initial_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr, historical_pdr_no_intervention):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Packet Delivery Ratio (PDR)")
    ax.set_title("Network Simulation")

    line, = ax.plot([], [], lw=2, label="Adaptive PDR")
    baseline_line, = ax.plot([], [], lw=2, linestyle='--', color='red', label="No Intervention Baseline")  # Baseline line

    ax.set_xlim(0, len(historical_pdr))
    ax.set_ylim(0, 1.1)

    attacked_nodes_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")
    disjoint_paths_text = ax.text(0.02, 0.9, "", transform=ax.transAxes, fontsize=10, va="top")
    current_path_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=10, va="top")

    def init():
        line.set_data([], [])
        baseline_line.set_data([], [])
        attacked_nodes_text.set_text("")
        disjoint_paths_text.set_text("")
        current_path_text.set_text("")
        return line, baseline_line, attacked_nodes_text, disjoint_paths_text, current_path_text

    def animate(t):
        line.set_data(range(t), historical_pdr[:t])
        baseline_line.set_data(range(t), historical_pdr_no_intervention[:t])  # Update baseline line

        current_path_text.set_text(f"Initial Path: {initial_path}")
        attacked_nodes_text.set_text(f"Attacked Nodes: {', '.join(map(str, attacked_nodes))}")
        disjoint_paths_text.set_text(f"Disjoint Paths: {', '.join(map(str, disjoint_paths))}")

        return line, baseline_line, attacked_nodes_text, disjoint_paths_text, current_path_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(historical_pdr) + 1, interval=250, blit=True, repeat=False)
    plt.legend()
    plt.show()

# Initialize node safety scores (higher score means less attacked, safer to use)
node_safety_scores = {}

def initialize_safety_scores(graph):
    """Initialize all nodes in the graph with a base safety score."""
    for node in graph.nodes():
        node_safety_scores[node] = 100  # Start all nodes with a neutral score

def update_safety_scores(attacked_nodes):
    """Decrease the safety score of attacked nodes and increase scores of non-attacked nodes over time."""
    for node in node_safety_scores:
        if node in attacked_nodes:
            node_safety_scores[node] -= 10  # Penalize attacked nodes
        else:
            node_safety_scores[node] += 2   # Reward non-attacked nodes for being 'safer'
        node_safety_scores[node] = max(0, min(node_safety_scores[node], 100))  # Clamp scores between 0 and 100

def find_path_with_safety_preference(graph, source, destination, min_length):
    """Find a path prioritizing nodes with higher safety scores using a modified version of Dijkstra's."""
    safety_weighted_graph = graph.copy()
    for u, v in safety_weighted_graph.edges():
        # Adjust edge weights by inversely relating them to node safety scores
        safety_weight = (node_safety_scores[u] + node_safety_scores[v]) / 2
        safety_weighted_graph[u][v]['weight'] /= safety_weight  # Higher score lowers effective weight

    # Use Dijkstra’s on this safety-weighted graph
    try:
        path = nx.shortest_path(safety_weighted_graph, source, destination, weight='weight')
        if len(path) >= min_length:
            return path
    except nx.NetworkXNoPath:
        pass
    return None

def simulate_network(graph, source, destination):
    """
    Simulate the network with sliding window PDR, attack response, and node safety learning.
    """
    # Initialize deque, PDR, and node safety scores
    packet_deque = deque(maxlen=max_pdr_window)
    pdr = 1.0
    initialize_safety_scores(graph)

    # Initial path and tracking
    current_path = nx.shortest_path(graph, source, destination)
    disjoint_paths = []
    attacked_nodes = set()
    used_paths = {tuple(current_path)}

    for t in range(attack_duration):
        # Send packets and handle attacks
        packets_sent = packets_per_second
        packets_received = packets_per_second
        num_attacks = attack_ratio_per_second
        attacked_nodes.clear()

        for _ in range(num_attacks):
            node_to_attack = random.choice([node for node in graph.nodes() if node != source and node != destination])
            attacked_nodes.add(node_to_attack)

        update_safety_scores(attacked_nodes)

        # Calculate packets received based on current path and attacked nodes
        for node in current_path:
            if node in attacked_nodes:
                packets_received -= int(packets_per_second * attack_percentage)

        packet_deque.append(packets_received / packets_sent)
        pdr = sum(packet_deque) / len(packet_deque)
        historical_pdr.append(pdr)

        # Output status
        print(f"Time: {t}, PDR: {pdr:.2f}, Current Path: {current_path}")

        # Check for path switching if PDR falls below threshold
        if pdr < pdr_threshold:
            disjoint_path = find_path_with_safety_preference(graph, source, destination, min_path_len)
            if disjoint_path and tuple(disjoint_path) not in used_paths:
                print(f"Switching to safer path: {disjoint_path}")
                current_path = disjoint_path
                disjoint_paths.append(disjoint_path)
                used_paths.add(tuple(disjoint_path))
            else:
                print("No suitable path with higher safety scores found.")
        time.sleep(0.03)

    return current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr

if __name__ == "__main__":
    file_name = "adjacency_60_0_7_1_updated.txt"
    file_path = os.path.join("Test_data", "MKU_files", "internetworks", file_name)

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

    # Read adjacency matrices and create graph
    network_graph = read_adjacency_matrices(file_path)

    # Run adaptive and baseline simulations
    current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr = simulate_network(network_graph, source, destination)
    historical_pdr_no_intervention = simulate_no_intervention(network_graph, source, destination, attack_duration, attack_ratio_per_second)

    # Visualize both simulations
    visualize_simulation(network_graph, source, destination, current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr, historical_pdr_no_intervention)
