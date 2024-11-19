import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random




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


# print nx.Graph 
def print_graph_matrices(graph):
    """
    Prints the adjacency matrices for edges, bandwidth, delay, and reliability directly from the graph structure.
    """
    num_nodes = graph.number_of_nodes()
    edges_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    bandwidth_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    delay_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    reliability_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for (i, j, attr) in graph.edges(data=True):
        edges_matrix[i, j] = 1
        edges_matrix[j, i] = 1  # Ensure symmetry for undirected graph
        bandwidth_matrix[i, j] = attr.get('bandwidth', 0)
        bandwidth_matrix[j, i] = attr.get('bandwidth', 0)
        delay_matrix[i, j] = attr.get('delay', 0)
        delay_matrix[j, i] = attr.get('delay', 0)
        reliability_matrix[i, j] = attr.get('reliability', 0)
        reliability_matrix[j, i] = attr.get('reliability', 0)

    # Print matrices
    print("Edges Matrix:\n", edges_matrix)
    print("\nBandwidth Matrix:\n", bandwidth_matrix)
    print("\nDelay Matrix:\n", delay_matrix)
    print("\nReliability Matrix:\n", reliability_matrix)
    
    
    
def find_shortest_disjoint_path(graph, source, destination, exclude_nodes):
    """
    Find the shortest node-disjoint path between source and destination, excluding the nodes in exclude_nodes.
    """
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()

    # Remove the excluded nodes from the copied graph
    graph_copy.remove_nodes_from(exclude_nodes)

    # Find the shortest path
    try:
        return nx.shortest_path(graph_copy, source, destination)
    except nx.NetworkXNoPath:
        return None


def find_shortest_disjoint_path_with_min_length(graph, source, destination, current_path, min_length, used_paths):
    """
    Find the shortest disjoint path between source and destination that is at least min_length long
    and has not been used before unless no other path is available.

    Parameters:
    graph (networkx.Graph): The input graph
    source (node): The source node
    destination (node): The destination node
    current_path (list): The current path that should be avoided
    min_length (int): The minimum length of the disjoint path
    used_paths (set): Set of previously used paths to avoid reuse if possible

    Returns:
    list or None: The shortest disjoint path that is at least min_length long, or None if no such path is found
    """
    # Create a copy of the graph to modify
    subgraph = graph.copy()

    # Remove all nodes from the subgraph that are in the current_path, except for source and destination
    for node in current_path:
        if node != source and node != destination:
            subgraph.remove_node(node)

    # Run the default path finding algorithm on the subgraph
    try:
        disjoint_path = nx.shortest_path(subgraph, source=source, target=destination)
    except nx.NetworkXNoPath:
        return None

    # Check if the disjoint path meets the minimum length requirement and has not been used before
    if len(disjoint_path) >= min_length and tuple(disjoint_path) not in used_paths:
        return disjoint_path
    elif len(disjoint_path) >= min_length:
        # If no other paths are available, allow reusing an old path
        for path in nx.all_shortest_paths(subgraph, source=source, target=destination):
            if len(path) >= min_length and tuple(path) not in used_paths:
                return path
        return disjoint_path  # Fallback to reusing an old path if necessary
    else:
        return None