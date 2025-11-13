import json
import pandas as pd
import sys
import os

def convert_runs_json_to_csv(json_file, output_folder):
    """
    Convert runs.json to workflows.csv format
    
    Parameters:
    - json_file: path to the input JSON file
    - output_folder: path to the output folder (will be created if it doesn't exist)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        runs = json.load(f)
    
    print(f"Loading {len(runs)} runs from JSON...")
    
    # Collect all unique parameter and metric names
    param_names = set()
    metric_names = set()
    
    # Process each run
    workflow_list = []
    
    for run in runs:
        # Initialize workflow dictionary with ID
        workflow = {'workflowId': run['id']}
        
        # Extract parameters
        for param in run['params']:
            param_name = param['name']
            param_value = param['value']
            
            # Convert parameter name format: fairness_method -> fairness method
            param_name_formatted = param_name.replace('_', ' ')
            param_names.add(param_name_formatted)
            
            # Try to convert numeric strings to numbers
            try:
                if '.' in str(param_value):
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
            except (ValueError, TypeError):
                pass  # Keep as string
            
            workflow[param_name_formatted] = param_value
        
        # Extract metrics
        for metric in run['metrics']:
            metric_name = metric['name']
            metric_value = metric['value']
            metric_names.add(metric_name)
            workflow[metric_name] = metric_value
        
        # Add status if available
        if 'status' in run:
            workflow['status'] = run['status']
        
        workflow_list.append(workflow)
    
    # Create DataFrame
    df = pd.DataFrame(workflow_list)
    df.drop(["status"], axis = 1, inplace = True)
    
    # Save to CSV in the output folder
    output_csv = os.path.join(output_folder, 'workflows.csv')
    df.to_csv(output_csv, index=False)
    
    # Save parameter names
    params_names_txt = os.path.join(output_folder, 'parameter_names.txt')
    with open(params_names_txt, 'w') as f:
        for name in sorted(param_names):
            f.write(name + '\n')
    
    # Save metric names
    metrics_names_txt = os.path.join(output_folder, 'metric_names.txt')
    with open(metrics_names_txt, 'w') as f:
        for name in sorted(metric_names):
            f.write(name + '\n')
    
    print(f"\n✓ Converted {len(df)} workflows")
    print(f"✓ Saved to '{output_csv}'")
    print(f"✓ Saved {len(param_names)} parameter names to '{params_names_txt}'")
    print(f"✓ Saved {len(metric_names)} metric names to '{metrics_names_txt}'")
    
    return df, param_names, metric_names

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 0_convert.py <input_json_file> <output_folder_name>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    try:
        df_converted, param_names, metric_names = convert_runs_json_to_csv(json_file, output_folder)
        print(f"\n✓ Conversion complete!")
        print(f"✓ All files saved to folder: '{output_folder}'")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
