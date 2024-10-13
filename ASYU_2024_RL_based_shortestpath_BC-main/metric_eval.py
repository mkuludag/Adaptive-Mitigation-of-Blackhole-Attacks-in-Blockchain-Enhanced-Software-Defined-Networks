import pandas as pd
import sys

def read_file(filename):
    """Reads a text file and returns a DataFrame."""
    try:
        df = pd.read_csv(filename, sep='\t')
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def calculate_metrics(df):
    """Calculates average bandwidth, path length, and execution time."""
    avg_bw = df['BW'].mean()
    avg_path_length = df['Num of Hops'].mean()
    avg_exec_time = df['Execution Time (seconds)'].mean()
    return avg_bw, avg_path_length, avg_exec_time

def compare_metrics(metrics1, metrics2):
    """Compares metrics of two files and returns comparison results."""
    comparison = {
        "BW": "File 1 has larger BW" if metrics1[0] > metrics2[0] else "File 2 has larger BW" if metrics1[0] < metrics2[0] else "Both have equal BW",
        "Path Length": "File 1 has lower Path Length" if metrics1[1] < metrics2[1] else "File 2 has lower Path Length" if metrics1[1] > metrics2[1] else "Both have equal Path Length",
        "Execution Time": "File 1 has lower Execution Time" if metrics1[2] < metrics2[2] else "File 2 has lower Execution Time" if metrics1[2] > metrics2[2] else "Both have equal Execution Time"
    }
    return comparison

def truncate_to_min_rows(df1, df2):
    """Truncates both DataFrames to the length of the smaller DataFrame."""
    min_rows = min(len(df1), len(df2))
    df1_truncated = df1.iloc[:min_rows]
    df2_truncated = df2.iloc[:min_rows]
    return df1_truncated, df2_truncated

def main(file1, file2=None):
    # Read files
    df1 = read_file(file1)
    
    if file2:
        df2 = read_file(file2)
    
    if df1 is not None and file2 is None:
        # Calculate metrics for single file
        metrics1 = calculate_metrics(df1)
        print(f"Metrics for {file1}:")
        print(f"Average BW: {metrics1[0]}")
        print(f"Average Path Length: {metrics1[1]}")
        print(f"Average Execution Time: {metrics1[2]} seconds\n")
    
    if df1 is not None and df2 is not None:
        # Truncate both DataFrames to the smaller length
        df1_truncated, df2_truncated = truncate_to_min_rows(df1, df2)

        # Calculate metrics for truncated DataFrames
        metrics1 = calculate_metrics(df1_truncated)
        metrics2 = calculate_metrics(df2_truncated)
        
        print(f"Metrics for {file1}:")
        print(f"Average BW: {metrics1[0]}")
        print(f"Average Path Length: {metrics1[1]}")
        print(f"Average Execution Time: {metrics1[2]} seconds\n")
        
        print(f"Metrics for {file2}:")
        print(f"Average BW: {metrics2[0]}")
        print(f"Average Path Length: {metrics2[1]}")
        print(f"Average Execution Time: {metrics2[2]} seconds\n")
        
        # Compare metrics
        comparison = compare_metrics(metrics1, metrics2)
        print("Comparison Results:")
        for metric, result in comparison.items():
            print(f"{metric}: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metric_evaluation.py <file1> [<file2>]")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2] if len(sys.argv) > 2 else None
    main(file1, file2)
