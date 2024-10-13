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

# Function to filter execution times between 100 and 200 milliseconds
def filter_execution_time_range(data):
    # Convert Execution Time (seconds) to milliseconds
    data["Execution Time (ms)"] = data["Execution Time (seconds)"] * 1000
    # Filter rows with execution time in the range of 100-200 ms
    #data = data[(data["Execution Time (ms)"] >= 100) & (data["Execution Time (ms)"] <= 200)]
    return data

# Function to calculate the percentage of RS for execution times at different intervals
def calculate_percentage_of_rs(data, time_intervals):
    percentages = []
    total_entries = len(data)
    for time in time_intervals:
        # Count entries where execution time is less than or equal to the current time
        viable_solutions = len(data[data["Execution Time (ms)"] <= time])
        # Calculate the percentage
        percentage = (viable_solutions / total_entries) * 100
        percentages.append(percentage)
    
    return percentages

# Function to extract relevant parts from file path
def extract_info_from_filename(file_path):
    # Extract just the file name without directories
    file_name = os.path.basename(file_path)
    # Extract key parts from the file name
    parts = file_name.split('_')
    return '_'.join(parts[2:]), '_'.join(parts[2:4])  # Returning key info from filename (like 'd_USNET.txt')

# Function to plot Percentage of RS vs Execution Time
def plot_percentage_of_rs(file1_data, file2_data, file1_label, file2_label, save_filename, loss):
    # Define execution time intervals (100 ms to 200 ms)
    time_intervals = np.arange(0, 5001, 50)  # Intervals of 25 ms

    # Calculate percentage of RS for both datasets
    file1_percentages = calculate_percentage_of_rs(file1_data, time_intervals)
    file2_percentages = calculate_percentage_of_rs(file2_data, time_intervals)

    # Plot settings for the line plot
    plt.figure(figsize=(10, 6))

    # Plot the data for both txt files
    plt.plot(time_intervals, file1_percentages, label=file1_label, marker='o', linestyle='-', color='b', linewidth=2)
    plt.plot(time_intervals, file2_percentages, label=file2_label, marker='s', linestyle='--', color='r', linewidth=2)

    # Label the axes
    plt.xlabel('Execution Time (ms)', fontsize=24)
    plt.ylabel('Percentage of RS (%)', fontsize=24)
    
    plt.tick_params(axis='both', which='major', labelsize=15)

    # # Add a title
    # if loss == 'result_d':
    #     plt.suptitle('Delay Based Loss Function', fontsize=26)
    # elif loss == 'result_bw':
    #     plt.suptitle('Bandwidth Based Loss Function', fontsize=26)
    # else:
    #     plt.suptitle('Bandwidth & Delay Based Loss Function', fontsize=26)
    # plt.title('Percentage of RS vs Execution Time', fontsize=15)

    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True)
    plt.savefig(f'figures/for_paper/percentage_of_rs_{save_filename}.pdf', bbox_inches='tight', format='pdf') #plt.savefig(f'figures/for_paper/percentage_of_rs_{save_filename}.png', bbox_inches='tight')
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
    #breakpoint()
    # Filter execution times within the range 100-200 ms
    file1_data = filter_execution_time_range(file1_data)
    file2_data = filter_execution_time_range(file2_data)

    # Extract key parts from file names to form the save file name
    file1_info, loss = extract_info_from_filename(file1_path)
    file2_info, _ = extract_info_from_filename(file2_path)

    # Combine the extracted parts to form the new file name
    save_filename = f'{file1_info}'  # Update if needed
    save_filename = lf + '_' + network

    # Plot the percentage of RS vs execution time
    plot_percentage_of_rs(file1_data, file2_data, "QoSCAPE", "GTN- QoSCAPE", save_filename, loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some input files.")
    parser.add_argument('lf', type=str, help='The link factor (e.g., bw)')
    parser.add_argument('network', type=str, help='The network name (e.g., US)')

    args = parser.parse_args()

    main(args.lf, args.network)
