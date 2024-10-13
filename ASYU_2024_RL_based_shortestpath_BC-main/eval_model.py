import time
import argparse
import torch
import os
import random
import numpy as np
from read_matrices import read_matrices
from Transactions import Transactions
from deep_rl_shortest import GraphTransformer, find_min_hop_for_current_as, read_transactions

# Function to load model weights
def load_model_weights(model, file_path):
    model.load_state_dict(torch.load(file_path))
    print(f"Model weights loaded from {file_path}")

# Function to evaluate the model
def evaluate_model(model, test_transactions, source, destination, bw, D, epsilon=0.05):
    s_cur = source
    path = [s_cur]
    visited = set()  # Set to track visited nodes
    visited.add(s_cur)
    total_hops = 0
    start_time = time.time()

    # Run the simulation using the trained model
    while s_cur != destination:
        s_next = epsilon_greedy(s_cur, model, epsilon, visited, D)
        
        if s_next is None:  # If no unvisited nodes are available, terminate the evaluation early
            print("No unvisited nodes available, terminating early")
            break
        
        hop = find_min_hop_for_current_as(test_transactions, s_cur, s_next, bw)
        total_hops += (hop.Hop if hop != -1 else 9999)  # Add hop count or penalty for invalid hops
        
        s_cur = s_next
        path.append(s_cur)
        visited.add(s_cur)  # Add the new state to the visited set
        
        if len(path) > len(D):  # Early termination condition if path length exceeds number of nodes
            print("Path length exceeded the number of nodes, terminating early")
            break

    end_time = time.time()
    execution_time = end_time - start_time

    # Log evaluation metrics
    print(f"Path: {path}")
    print(f"Total Hops: {total_hops}")
    print(f"Execution Time: {execution_time} seconds")

    return path, total_hops, execution_time



def epsilon_greedy(s_curr, model, epsilon, visited, D):
    potential_next_states = np.where(np.array(D[s_curr]) > 0)[0]
    
    # Remove already visited nodes from potential next states
    unvisited_states = [state for state in potential_next_states if state not in visited]
    
    if not unvisited_states:
        return random.choice(potential_next_states)  # If no unvisited states, choose from all potential states
    
    if random.random() > epsilon:  # greedy
        q_values = model(torch.FloatTensor(D[s_curr]).unsqueeze(0))  # Get Q-values from the model
        s_next = unvisited_states[torch.argmax(q_values[0][unvisited_states]).item()]
    else:  # random select
        s_next = random.choice(unvisited_states)
    
    return s_next

# Inference function to run the model simulation with CLI arguments
def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Source node", type=int, default=0)
    parser.add_argument("-d", "--destination", help="Destination node", type=int, default=5)
    parser.add_argument("-b", "--bandwidth", help="Bandwidth demand", type=int, default=10)
    parser.add_argument("-m", "--model_path", help="Path to saved model weights", type=str, default="graph_transformer_weights.pth")
    args = parser.parse_args()
    
    network = 'NSFNET'
    network = 'USNET'
    
    # Load the adjacency matrix and transactions for testing
    cur_path = os.getcwd()
    file_path = os.path.join(cur_path, network, 'network', 'adjacency_14_0_1_2_updated.txt')
    D = read_matrices(file_path)
    num_nodes = len(D)

    test_transactions_file = os.path.join(cur_path, network, 'transactions', 'transactions_nsfnet_5nodes.txt')
    test_transactions = read_transactions(test_transactions_file)

    # Initialize the model
    model = GraphTransformer(input_dim=num_nodes, output_dim=num_nodes)

    # Load pretrained model weights
    load_model_weights(model, args.model_path)

    # Run the evaluation
    evaluate_model(model, test_transactions, args.source, args.destination, args.bandwidth, D)

if __name__ == "__main__":
    run_inference()
