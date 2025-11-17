import pandas as pd 
import json

def read_data(csv_path, cluster_id, rule):
    df = pd.read_csv(csv_path)
    df = filter_data(df, cluster_id=cluster_id, rule=rule)
    return df
    

def filter_data(df, cluster_id=None, rule=None):
    """
    Filter the DataFrame based on cluster_id and rule.
    """
    if cluster_id is not None:
        df = df[df['cluster'] == cluster_id]
    
    if rule:
        try:
            df = df.query(rule)
        except Exception as e:
            print(f"Error applying rule '{rule}': {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    return df


def convert_rule_syntax(rule):
    """
    Convert rule syntax from JSON format to pandas query format.
    - Replace = with ==
    - Replace IN with in
    - Replace {...} with [...]
    - Add backticks around column names with spaces
    """
    # Replace single = with == for equality (but not <= or >=)
    import re
    rule = re.sub(r'(?<![<>!])=(?!=)', '==', rule)
    
    # Replace IN with in
    rule = rule.replace(' IN ', ' in ')
    
    # Replace {...} with [...]
    rule = rule.replace('{', '[').replace('}', ']')
    
    # Add backticks around common column names with spaces
    column_mappings = {
        'fairness_method': '`fairness method`',
        'max_depth': '`max depth`',
        'n_estimators': '`n estimators`',
        'model_type': '`model type`',
        'random_state': '`random state`',
    }
    
    for old_col, new_col in column_mappings.items():
        rule = rule.replace(old_col, new_col)
    
    return rule


def process_clusters(json_path, csv_path):
    """
    Read the clusters JSON file and process each cluster's rules.
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        clusters_data = json.load(f)
    
    # Load the CSV once
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")
    
    results = []
    
    # Iterate through each cluster
    for cluster_key, cluster_info in clusters_data.items():
        cluster_id = cluster_info['cluster_id']
        decision_rules = cluster_info.get('decision_tree_rules', [])
        
        print(f"{'='*80}")
        print(f"Processing Cluster {cluster_id}")
        print(f"{'='*80}")
        
        # Iterate through each rule in the cluster
        for idx, rule_info in enumerate(decision_rules):
            rule = rule_info['rule']
            f1_score = rule_info.get('f1_score', 'N/A')
            precision = rule_info.get('precision', 'N/A')
            recall = rule_info.get('recall', 'N/A')
            n_workflows = rule_info.get('n_workflows_in_cluster', 'N/A')
            
            print(f"\nRule {idx + 1}/{len(decision_rules)}:")
            print(f"  Original Rule: {rule}")
            
            # Convert rule syntax
            converted_rule = convert_rule_syntax(rule)
            print(f"  Converted Rule: {converted_rule}")
            
            # Filter data
            filtered_df = read_data(csv_path, cluster_id, converted_rule)
            
            print(f"  Expected workflows: {n_workflows}")
            print(f"  Actual filtered rows: {len(filtered_df)}")
            print(f"  F1 Score: {f1_score}")
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            
            # Store results
            results.append({
                'cluster_id': cluster_id,
                'rule_index': idx,
                'rule': rule,
                'converted_rule': converted_rule,
                'expected_workflows': n_workflows,
                'actual_rows': len(filtered_df),
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall
            })
            
            # Show sample data if rows exist
            if len(filtered_df) > 0 and 'Precision' in filtered_df.columns and 'Recall' in filtered_df.columns and 'F1 Score' in filtered_df.columns:
                print(f"\n  Statistics:")
                means = filtered_df[["Precision", "Recall", "F1 Score"]].mean()
                stds = filtered_df[["Precision", "Recall", "F1 Score"]].std()
                variances = filtered_df[["Precision", "Recall", "F1 Score"]].var()
                
                print(f"    Precision: mean={means['Precision']:.4f}, std={stds['Precision']:.4f}, var={variances['Precision']:.6f}")
                print(f"    Recall:    mean={means['Recall']:.4f}, std={stds['Recall']:.4f}, var={variances['Recall']:.6f}")
                print(f"    F1 Score:  mean={means['F1 Score']:.4f}, std={stds['F1 Score']:.4f}, var={variances['F1 Score']:.6f}")
                print(f"    Range (mean±std):")
                print(f"      Precision: ({means['Precision']-stds['Precision']:.4f}, {means['Precision']+stds['Precision']:.4f})")
                print(f"      Recall:    ({means['Recall']-stds['Recall']:.4f}, {means['Recall']+stds['Recall']:.4f})")
                print(f"      F1 Score:  ({means['F1 Score']-stds['F1 Score']:.4f}, {means['F1 Score']+stds['F1 Score']:.4f})")


        
        print(f"\n")
    
    return results


if __name__ == "__main__":
    json_path = './data/workflows/clusters_comprehensive_insights.json'
    csv_path = './data/workflows/workflows_clustered.csv'
    
    results = process_clusters(json_path, csv_path)
    
    print(f"\n{'='*80}")
    print(f"Summary: Processed {len(results)} rules across all clusters")
    print(f"{'='*80}")