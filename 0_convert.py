import json
import pandas as pd

def convert_runs_json_to_csv(json_file, output_csv):
    """
    Convert runs.json to workflows.csv format
    
    Parameters:
    - json_file: path to the input JSON file
    - output_csv: path to the output CSV file
    """
    
    # Load JSON data
    with open(json_file, 'r') as f:
        runs = json.load(f)
    
    print(f"Loading {len(runs)} runs from JSON...")
    
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
            workflow[metric_name] = metric_value
        
        # Add status if available
        if 'status' in run:
            workflow['status'] = run['status']
        
        workflow_list.append(workflow)
    
    # Create DataFrame
    df = pd.DataFrame(workflow_list)
    df.drop(["status"], axis = 1, inplace = True)
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Converted {len(df)} workflows")
    print(f"✓ Saved to '{output_csv}'")
    
    return df

# Usage
df_converted = convert_runs_json_to_csv('runs.json', 'workflows.csv')
