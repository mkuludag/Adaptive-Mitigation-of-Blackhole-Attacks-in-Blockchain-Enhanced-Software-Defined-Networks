import string
from argparse import ArgumentParser
import random
import time
import os
import numpy as np
from Transactions import Transactions
from read_matrices import read_matrices
import torch
from deep_rl_shortest import GraphTransformer, find_min_hop_for_current_as, read_transactions

random.seed(100)
np.random.seed(50)

# Load pre-trained model
def load_model(num_nodes, weights_path):
    model = GraphTransformer(input_dim=num_nodes, output_dim=num_nodes)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# Function to evaluate the model with delay calculation
def evaluate_model(model, test_transactions, source, destination, bw, D, epsilon=0.05):
    s_cur = source
    path = [s_cur]
    visited = set()  # Set to track visited nodes
    visited.add(s_cur)
    total_hops = 0
    total_delay = 0.0  # Initialize total delay
    start_time = time.time()

    # Run the simulation using the trained model
    while s_cur != destination:
        s_next = epsilon_greedy(s_cur, model, epsilon, visited, D)
        
        if s_next is None:  # If no unvisited nodes are available, terminate the evaluation early
            print("No unvisited nodes available, terminating early")
            break
        
        hop = find_min_hop_for_current_as(test_transactions, s_cur, s_next, bw)
        total_hops += (hop.Hop if hop != -1 else 9999)  # Add hop count or penalty for invalid hops
        total_delay += (hop.Delay if hop != -1 else 9999)  # Add delay or penalty for invalid delay
        
        s_cur = s_next
        path.append(s_cur)
        visited.add(s_cur)  # Add the new state to the visited set
        
        if len(path) > len(D):  # Early termination condition if path length exceeds number of nodes
            print("Path length exceeded the number of nodes, terminating early")
            break

    end_time = time.time()
    execution_time = end_time - start_time

    # Log evaluation metrics
    #print(f"Path: {path}")
    #print(f"Total Hops: {total_hops}")
    #print(f"Total Delay: {total_delay}")
    #print(f"Execution Time: {execution_time} seconds")

    return path, total_hops, total_delay, execution_time

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


def write_to_file(numofNodes, bw, num_of_hops, total_delay, path, execution_time, txt_file):
    with open(txt_file, 'a') as file:
        # Header only written in the first line
        if file.tell() == 0:
            file.write("Num of Nodes\tBW\tDelay\tNum of Hops\tPath\tExecution Time (seconds)\n")
        file.write(f"{numofNodes}\t{bw}\t{total_delay}\t{num_of_hops}\t{path}\t{execution_time}\n")

        


if __name__ == '__main__': 
    parser = ArgumentParser()
    
    network = 'NSFNET'
    network = 'USNET'
    txt_file = f"results_d/pretrained_rl_result_d_" + network + ".txt"
    
    parser.add_argument("-w", "--weights", help="Path to pre-trained model weights", type=str, required=True)
    args = parser.parse_args()

    cur_path = os.getcwd()
    file_path = os.path.join(cur_path, network, 'network', 'adjacency_24_0_1_1_updated.txt') #adjacency_14_0_1_2_updated.txt
    matrices = read_matrices(file_path)
    D = matrices
    num_nodes = len(D)

    model = load_model(num_nodes, args.weights)

    number = [5, 6, 7, 8, 9, 10]
    for i in number:
        # Read transactions
        transactions = f"transactions_usnet_{i}nodes.txt" #f"transactions_nsfnet_{i}nodes.txt"
        file_path = os.path.join(cur_path, network, 'transactions', transactions)
        all_transactions = read_transactions(file_path)

        reqs = f"requests_usnet_{i}nodes.txt"  #f"requests_nsfnet_{i}nodes.txt"
        req_path = os.path.join(cur_path, network, 'requests', reqs)
        with open(req_path, 'r') as file:
            headers = file.readline().strip().split('\t')
            for line in file:
                source_as, destination_as, bandwidth = line.strip().split('\t')
                bwDemand = [1, 5, 10, 15, 20, 25]

                for bw in bwDemand:
                    start_time = time.time()
                    path, num_of_hops, total_delay, execution_time = evaluate_model(model, all_transactions, int(source_as), int(destination_as), bw, D)
                    write_to_file(i, bw, num_of_hops, total_delay, path, execution_time, txt_file)
