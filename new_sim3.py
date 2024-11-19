import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from queue import PriorityQueue
import time
import os

# Import your helpers
from new_helpers import read_adjacency_matrices, print_graph_matrices, find_shortest_disjoint_path, find_shortest_disjoint_path_with_min_length

# Simulation parameters
packets_per_second = 10
pdr_threshold = 0.95
global_min_path_len = 1  # Minimum path length
attack_duration = 50  # Timesteps
attack_ratio_per_second = 2  # Number of nodes attacked per second
attack_percentage = 0.2  # Percentage of packets dropped during attack
max_pdr_window = 50 # max number of elements in deque which stores the packets being recieved

historical_pdr = []

# RL and Deep RL (Graph Transformer) Network Models

# Define your deep RL model (Graph Transformer or DQN-based)
class GraphTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphTransformer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Network (DQN) for RL-based Path Selection
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the reinforcement learning agent for path selection
class RLAgent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.model = DQN(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.1  # Exploration rate

    def select_action(self, state):
        # Exploration vs Exploitation tradeoff
        if random.random() < self.epsilon:
            return random.randint(0, len(state) - 1)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        with torch.no_grad():
            target = reward + (1 - done) * 0.99 * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32)))
        
        self.optimizer.zero_grad()
        prediction = self.model(torch.tensor(state, dtype=torch.float32))[action]
        loss = self.loss_fn(prediction, target)
        loss.backward()
        self.optimizer.step()

# Visualization function
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

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(historical_pdr) + 1, interval=250, blit=True, repeat=False)
    plt.show()

# Simulation Function with Deep RL and RL
def simulate_network_with_rl(graph, source, destination, agent):
    """
    Simulate the network with RL and Deep RL path selection
    """
    packet_deque = deque(maxlen=max_pdr_window)
    historical_pdr = []

    current_path = nx.shortest_path(graph, source, destination)
    disjoint_paths = []
    attacked_nodes = set()
    used_paths = {tuple(current_path)}  # Store initial path as used

    for t in range(attack_duration):
        packets_sent = packets_per_second
        packets_received = packets_per_second

        num_attacks = attack_ratio_per_second
        for _ in range(num_attacks):
            node_to_attack = random.choice([node for node in graph.nodes() if node != source and node != destination])
            attacked_nodes.add(node_to_attack)

        for node in current_path:
            if node in attacked_nodes:
                packets_received = int(packets_received - packets_per_second * attack_percentage)

        packet_deque.append(packets_received / packets_sent)
        pdr = sum(packet_deque) / len(packet_deque)
        historical_pdr.append(pdr)

        state = [pdr]  # Define the state as PDR for RL
        action = agent.select_action(state)  # Select action based on RL model
        reward = pdr  # You can modify the reward function based on your needs
        next_state = [pdr]  # Next state can be defined as the updated PDR or other features
        
        agent.train(state, action, reward, next_state, done=False)

        if pdr < pdr_threshold:
            disjoint_path = find_shortest_disjoint_path_with_min_length(graph, source, destination, current_path, global_min_path_len, used_paths)
            if disjoint_path:
                current_path = disjoint_path
                disjoint_paths.append(disjoint_path)
                used_paths.add(tuple(disjoint_path))

    return current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr

if __name__ == "__main__":
    file_name = "adjacency_60_0_7_1_updated.txt"
    file_path = os.path.join("Test_data", "MKU_files", "internetworks", file_name)

    # Parse command-line arguments
    source = 3
    destination = 59

    # Read the adjacency matrices and create the graph
    network_graph = read_adjacency_matrices(file_path)

    # Create RL agent with input size, hidden layer size, and output size
    agent = RLAgent(input_size=1, hidden_size=64, output_size=len(network_graph.nodes))

    # Run the simulation with RL-based decision making
    current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr = simulate_network_with_rl(network_graph, source, destination, agent)

    # Visualize the simulation results
    visualize_simulation(network_graph, source, destination, current_path, disjoint_paths, packet_deque, attacked_nodes, historical_pdr)
