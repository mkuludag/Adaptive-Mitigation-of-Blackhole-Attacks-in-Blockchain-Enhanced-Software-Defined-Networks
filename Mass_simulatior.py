import os
import subprocess

# Directory containing the adjacency matrix files
directory = 'Test_data/MKU_files/internetworkBlackHole/internetworkBlackHole'

# Get a list of all adjacency matrix files in the directory
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]

# Iterate over each file and call the dom_sim2.py script
for file_path in file_paths:
    print(f"Running simulation for file: {file_path}")
    subprocess.run(["python", "dom_sim2.py", "--file", file_path])
