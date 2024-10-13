import subprocess

def run_analysis_scripts():
    # Define the loss functions and networks
    loss_functions = ['bw', 'd', 'd_bw']
    networks = ['US', 'NSF']

    # List of scripts to run
    scripts = ['Perc_RS_figure.py', 'Exec_t_figure.py']

    # Iterate over all combinations of loss functions and networks
    for lf in loss_functions:
        for network in networks:
            for script in scripts:
                # Construct the command to run the analysis script
                command = ['python3', script, lf, network]
                try:
                    # Run the command and wait for it to complete
                    print(f"Running: {' '.join(command)}")
                    result = subprocess.run(command, check=True)
                    print(f"Finished: {' '.join(command)} with exit code {result.returncode}")
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while running: {e}")

if __name__ == "__main__":
    run_analysis_scripts()
