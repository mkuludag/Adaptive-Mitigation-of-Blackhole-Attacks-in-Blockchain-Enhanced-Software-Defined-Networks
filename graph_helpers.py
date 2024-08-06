import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def read_transactions(file_path):
    graph = nx.Graph()
    with open(file_path, 'r') as file:
        next(file)  # Skip header line
        intra_domain_edges = []
        inter_domain_edges = []
        for line in file:
            if line.strip() == "":
                continue
            parts = line.split('\t')
            transaction_id = int(parts[0])
            as_id = int(parts[1])  # AS ID
            ingress = int(parts[2])  # Ingress node
            egress = int(parts[3])  # Egress node
            pathlet_id = int(parts[4])  # Pathlet ID
            bandwidth = float(parts[5])  # Bandwidth
            delay = float(parts[6])  # Delay
            reliability = float(parts[7])  # Reliability
            status = parts[8].strip().lower() == 'true'  # Status (convert to boolean)
            
            # Extract full path nodes
            full_path_str = parts[9].strip().strip('[]')
            node_strs = full_path_str.split(',')
            full_path = [int(node_str.strip()) for node_str in node_strs]
            
            edge_list = intra_domain_edges if as_id != -1 else inter_domain_edges
            # Add edges for each hop in the full path
            for i in range(len(full_path) - 1):
                node1 = full_path[i]
                node2 = full_path[i + 1]
                edge_list.append((node1, node2, {'bandwidth': bandwidth, 'delay': delay, 'reliability': reliability, 'status': status}))

    # Add intra-domain edges first
    graph.add_edges_from(intra_domain_edges)
    # Set 'safe' attribute for each node
    for node in graph.nodes:
        graph.nodes[node]['safe'] = True

    # Add inter-domain edges after
    graph.add_edges_from(inter_domain_edges)
        
    return graph

def visualize_graph(graph, shortest_path=None, show_edge_labels=False, show_cross_domain_edges=False):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Separate same-domain and cross-domain edges
    same_domain_edges = [(u, v) for u, v, d in graph.edges(data=True) if int(u / 1000) == int(v / 1000)]
    cross_domain_edges = [(u, v) for u, v, d in graph.edges(data=True) if int(u / 1000) != int(v / 1000)]

    # Create a graph with only same-domain edges to determine node positions
    same_domain_graph = nx.Graph()
    same_domain_graph.add_nodes_from(graph.nodes(data=True))
    same_domain_graph.add_edges_from(same_domain_edges)

    # Determine positions using the intra-domain graph
    pos = nx.spring_layout(same_domain_graph)

    # Draw the graph with same-domain edges
    nx.draw(same_domain_graph, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, ax=ax)
    nx.draw_networkx_edges(same_domain_graph, pos, edge_color='blue', width=2)

    # Optionally add cross-domain edges to the graph and draw them
    if show_cross_domain_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=cross_domain_edges, edge_color='red', width=2, style='dashed')

    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(same_domain_graph, 'bandwidth')  # Using bandwidth for edge labels
        nx.draw_networkx_edge_labels(same_domain_graph, pos, edge_labels=edge_labels, ax=ax)

    # Highlight the shortest path
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_nodes(same_domain_graph, pos, nodelist=shortest_path, node_color='orange')
        nx.draw_networkx_edges(same_domain_graph, pos, edgelist=path_edges, edge_color='orange', width=2)

    plt.title("Network Graph with Same-domain (blue) and Cross-domain (red) Edges")
    plt.show()




# def visualize_graph(graph, shortest_path):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     pos = nx.spring_layout(graph)
#     nx.draw(graph, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, ax=ax)
#     edge_labels = nx.get_edge_attributes(graph, 'bandwidth') # just use bandwith for now for edges
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

#     # Highlight the shortest path
#     if shortest_path:
#         path_edges = list(zip(shortest_path, shortest_path[1:]))
#         nx.draw_networkx_nodes(graph, pos, nodelist=shortest_path, node_color='orange')
#         nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='orange', width=2)

#     plt.show()

def find_shortest_path(graph, start_node, destination_node):
    safe_graph = graph.copy()
    unsafe_nodes = [node for node in graph.nodes if not graph.nodes[node].get('safe', True)]
    safe_graph.remove_nodes_from(unsafe_nodes)
    
    try:
        #return nx.shortest_path(safe_graph, source=start_node, target=destination_node, weight='weight', method='dijkstra')
        return nx.shortest_path(safe_graph, source=start_node, target=destination_node, weight='bandwidth', method='dijkstra')

    except nx.NetworkXNoPath:
        return None

def save_adjacency_matrix(graph, file_path):
    with open(file_path, 'w') as f:
        for line in nx.generate_adjlist(graph):
            f.write(line + '\n')

def deactivate_node(graph, node):
    if node in graph.nodes:
        graph.nodes[node]['safe'] = False

def calculate_packet_delivery_ratio(graph, paths):
    total_pdr = 0
    num_paths = len(paths)
    
    for i, path in enumerate(paths):
        path_pdr = 100
        
        # Decrease PDR for each unsafe node in the path
        for node in path:
            if not graph.nodes[node].get('safe', True):
                path_pdr -= random.uniform(1.5, 4.5)  # Random drop between 1.5% and 3.5%
        
        # Increase PDR slightly based on path index (optional)
        additional_pdr = 2 * i
        path_pdr = min(100, path_pdr + additional_pdr)
        
        # Apply a slight random probability drop
        random_drop = random.uniform(0.1, 0.5)  # Random drop between 0.1% and 0.5%
        path_pdr -= random_drop
        
        # Ensure PDR does not fall below 0%
        path_pdr = max(0, path_pdr)
        
        total_pdr += path_pdr
    
    return total_pdr / num_paths if num_paths else 100
    
    # Return average PDR across all paths, defaulting to 100 if no paths
    return total_pdr / num_paths if num_paths else 100


def find_node_disjoint_path(graph, start_node, destination_node, exclude_nodes):
    # Create a subgraph that excludes the specified nodes, except start and end nodes
    safe_graph = graph.copy()
    unsafe_nodes = [node for node in graph.nodes if not graph.nodes[node].get('safe', True)]
    safe_graph.remove_nodes_from(unsafe_nodes)
    
    # Ensure start and end nodes are not excluded
    exclude_nodes = [node for node in exclude_nodes if node not in [start_node, destination_node]]
    safe_graph.remove_nodes_from(exclude_nodes)
    
    try:
        # Use Dijkstra's algorithm to find the shortest path in the safe graph
        #return nx.shortest_path(safe_graph, source=start_node, target=destination_node, weight='weight', method='dijkstra')
        return nx.shortest_path(safe_graph, source=start_node, target=destination_node, weight='bandwidth', method='dijkstra')

    except nx.NetworkXNoPath:
        return None


    
    

def add_node_disjoint_path(current_paths, graph, start_node, destination_node):
    # Collect all nodes that are part of the existing paths
    excluded_nodes = set(node for path in current_paths for node in path)
    
    # Try to find a new node-disjoint path
    new_path = find_node_disjoint_path(graph, start_node, destination_node, excluded_nodes)
    
    # If a new node-disjoint path is found and it's unique, add it to the list
    if new_path and new_path not in current_paths:
        current_paths.append(new_path)
    else:
        # If no new disjoint path is found, find any path, even including unsafe nodes
        try:
            all_nodes_to_exclude = excluded_nodes - {start_node, destination_node}
            all_path = nx.shortest_path(graph, source=start_node, target=destination_node, weight='weight', method='dijkstra')
            # Ensure the new path doesn't include the excluded nodes, unless it's the start or destination
            if all_path not in current_paths:
                current_paths.append(all_path)
            print("Added a path with unsafe nodes")
        except nx.NetworkXNoPath:
            print("No path found to add :(")

    return current_paths



# Domain Algorithms: 

def read_adjacency_matrices(file_path):
    """
    Reads four NxN adjacency matrices from a file and creates a graph.
    The file contains four NxN matrices: edges, bandwidth, delay, and reliability.

    Parameters:
    - file_path: Path to the file containing the adjacency matrices.

    Returns:
    - graph: A NetworkX graph object with edge attributes.
    """
    # Read the entire file into a 2D array, handling whitespace
    data = np.loadtxt(file_path, delimiter='\t', dtype=int)
    
    # Determine the number of nodes (N)
    num_nodes = data.shape[1]

    # Extract the first 60x60 matrix for edges
    edges_matrix = data[:num_nodes, :num_nodes]
    bandwidth_matrix = data[num_nodes:2*num_nodes, :num_nodes]
    delay_matrix = data[2*num_nodes:3*num_nodes, :num_nodes]
    reliability_matrix = data[3*num_nodes:4*num_nodes, :num_nodes]

    # Create an empty graph
    graph = nx.Graph()
    
    # Add nodes
    graph.add_nodes_from(range(num_nodes))

    # Add edges and attributes based on matrices
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if edges_matrix[i, j] == 1:
                graph.add_edge(i, j, bandwidth=bandwidth_matrix[i, j],
                               delay=delay_matrix[i, j], reliability=reliability_matrix[i, j])

    for node in graph.nodes:
        graph.nodes[node]['safe'] = True  # Example attribute
    
    return graph


def visualize_domains(graph, final_path=None, num_nodes=60):
    """
    Visualizes the domain graph with nodes and edges.
    Highlights the final path if provided.

    Parameters:
    - graph: A NetworkX graph object representing the domain graph.
    - final_path: List of nodes representing the final path to highlight (optional).
    """
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Define node positions using a layout algorithm
    pos = nx.spring_layout(graph, seed=42)  # Fix seed for reproducibility

    # Draw the entire graph with all nodes and edges
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, ax=ax, edge_color='black')

    # Draw edges only (this is redundant because the edges are already drawn above)
    # nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='black')

    # Highlight the final path if provided
    if final_path:
        path_edges = list(zip(final_path, final_path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=final_path, node_color='orange')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='orange', width=2)

    # Add labels for edges if needed
    # edge_labels = nx.get_edge_attributes(graph, 'bandwidth')  # Example for edge labels
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    plt.title("Domain Graph Visualization with Final Path Highlighted")
    filename = f'results/graph_{num_nodes}.png'
    plt.savefig(filename)
    plt.show()