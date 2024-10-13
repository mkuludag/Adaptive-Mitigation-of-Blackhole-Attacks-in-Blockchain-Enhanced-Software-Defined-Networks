import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Function to load data from a txt file
def load_data(file_path):
    # Read the txt file into a pandas DataFrame, assuming tab-separated columns
    data = pd.read_csv(file_path, sep='\t')
    return data

# Function to filter out execution times above 500 milliseconds
def filter_outliers(data):
    # Convert Execution Time (seconds) to milliseconds and filter out rows where time is > 500 ms
    #data = data[data["Execution Time (seconds)"] * 1000 <= 500]
    data["Execution Time (ms)"] = data["Execution Time (seconds)"] * 1000

    # Set values greater than 500 ms to 500 ms
    #data.loc[data["Execution Time (ms)"] > 500, "Execution Time (ms)"] = 500
    return data

# Function to calculate medians for execution times grouped by "Num of Nodes"
def calculate_medians(data):
    # Group by "Num of Nodes" and calculate the median of "Execution Time (seconds)"
    median_data = data.groupby("Num of Nodes")["Execution Time (seconds)"].mean().reset_index()
    median_data["Execution Time (ms)"] = median_data["Execution Time (seconds)"] * 1000  # Convert to ms
    return median_data

# Function to extract relevant parts from file path
def extract_info_from_filename(file_path):
    # Extract just the file name without directories
    file_name = os.path.basename(file_path)
    # Extract key parts from the file name
    parts = file_name.split('_')
    return '_'.join(parts[2:]), '_'.join(parts[2:4])  # Returning key info from filename (like 'd_USNET.txt')

# Function to plot the median execution times
def plot_median_execution_time(file1_data, file2_data, file1_label, file2_label, save_filename, loss):
    # Calculate the median execution times for both files
    file1_medians = calculate_medians(file1_data)
    file2_medians = calculate_medians(file2_data)

    # Extract relevant columns from both datasets
    switches_per_node_1 = file1_medians["Num of Nodes"]
    execution_time_1 = file1_medians["Execution Time (ms)"]

    switches_per_node_2 = file2_medians["Num of Nodes"]
    execution_time_2 = file2_medians["Execution Time (ms)"]

    # Plot settings for the line plot
    plt.figure(figsize=(10, 6))

    # Plot the data for both txt files
    plt.plot(switches_per_node_1, execution_time_1, label=file1_label, marker='o', linestyle='-', color='b', linewidth=2)
    plt.plot(switches_per_node_2, execution_time_2, label=file2_label, marker='s', linestyle='--', color='r', linewidth=2)

    # Label the axes
    plt.xlabel('Num of Nodes per ISP', fontsize=24)
    plt.ylabel('Median PST (ms)', fontsize=24)
    
    plt.tick_params(axis='both', which='major', labelsize=15)

    # # Add a title
    # if loss == 'result_d':
    #     plt.suptitle('Delay Based Loss Function', fontsize=26) 
    # elif loss == 'result_bw':
    #     plt.suptitle('Bandwidth Based Loss Function', fontsize=26) 
    # else:
    #     plt.suptitle('Bandwidth & Delay Based Loss Function', fontsize=26) 
    # plt.title('Median Execution Time vs Num of Switches per Node', fontsize=15)
    

    # Add a legend
    plt.legend(loc='upper left', fontsize=20)

    # Show the grid
    plt.grid(True)
    plt.savefig(f'figures/for_paper/median_execution_time_{save_filename}.pdf', bbox_inches='tight', format='pdf')
    #plt.show()
    plt.close()
    

# Main function to read files and plot the data
def main(lf, network):
    # File paths (update these to your actual file paths)
    #lf = 'bw'
    #network = 'US' 
    network += 'NET'
    file1_path = f'results_d/rl_result_{lf}_{network}.txt'
    file2_path = f'results_d/pretrained_rl_result_{lf}_{network}.txt'

    # Load data from the two files
    file1_data = load_data(file1_path)
    file2_data = load_data(file2_path)

    # Filter outliers (times > 500 ms)
    file1_data = filter_outliers(file1_data)
    file2_data = filter_outliers(file2_data)

    # Extract key parts from file names to form the save file name
    file1_info, loss = extract_info_from_filename(file1_path)
    file2_info, _ = extract_info_from_filename(file2_path)

    # Combine the extracted parts to form the new file name
    #save_filename = f'{file1_info}' #_{file2_info}'
    save_filename = lf + '_' + network

    # Plot the median execution times
    plot_median_execution_time(file1_data, file2_data, "QoSCAPE", "GTN- QoSCAPE", save_filename, loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input files.")
    parser.add_argument('lf', type=str, help='The link factor (e.g., bw)')
    parser.add_argument('network', type=str, help='The network name (e.g., US)')

    args = parser.parse_args()

    main(args.lf, args.network)
