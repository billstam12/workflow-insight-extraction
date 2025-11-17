"""
Validate Cluster Rules - Does This Actually Work?

Imagine you clustered 100+ ML workflows and got rules like "use max_depth > 11.5 for high accuracy."
But how do you know it's real and not just random patterns?

This script answers 3 questions:
1. DOES IT WORK? - Apply each rule to all workflows. Do they actually land in the right cluster? (Precision check)
2. IS IT RELIABLE? - Look at workflows matching the rule. Do they all perform similarly, or is there huge variance? (CV < 0.1 = reliable)
3. WHAT DO I GET? - If I use this rule, what exact performance should I expect? (Mean ± Std for every metric)

Think of it like this: Your friend says "restaurants on Main St are always good."
You validate by: (1) Checking if Main St restaurants are actually rated high, (2) Seeing if they're consistently good (not 5-star one day, 1-star next), (3) Calculating the average rating ± variance.

Same logic here, but for ML workflow configurations instead of restaurants.

Usage: python 4_validate_cluster_rules.py data/workflows
Output: JSON/CSV with validated rules + plots showing which rule to use for high Accuracy/Recall/etc.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys


def parse_rule_condition(rule_str, row):
    """
    Parse a rule string and check if a workflow matches it.
    
    Args:
        rule_str: Rule string like "max_depth > 11.5 and fairness_method = 'none'"
        row: DataFrame row representing a workflow
    
    Returns:
        Boolean indicating if the workflow matches the rule
    """
    # Replace column names with actual values from the row
    rule_eval = rule_str
    
    # Handle IN statements first
    import re
    in_pattern = r"(\w+)\s+IN\s+\{([^}]+)\}"
    for match in re.finditer(in_pattern, rule_str):
        col_name = match.group(1)
        values_str = match.group(2)
        # Parse the set values
        values = [v.strip().strip("'\"") for v in values_str.split(',')]
        actual_value = str(row[col_name])
        is_in = actual_value in values
        rule_eval = rule_eval.replace(match.group(0), str(is_in))
    
    # Convert SQL-style = to Python ==
    # But preserve <=, >=, !=
    rule_eval = re.sub(r'([^<>=!])\s*=\s*([^=])', r'\1 == \2', rule_eval)
    
    # Handle simple comparisons
    for col in row.index:
        if col in rule_eval:
            val = row[col]
            # Quote strings for evaluation
            if isinstance(val, str):
                val_str = f"'{val}'"
            else:
                val_str = str(val)
            # Replace column name with value (word boundary to avoid partial matches)
            rule_eval = re.sub(rf'\b{col}\b', val_str, rule_eval)
    
    try:
        return eval(rule_eval)
    except Exception as e:
        # If evaluation fails, return False
        print(f"Warning: Failed to evaluate rule: {rule_eval}")
        print(f"  Error: {e}")
        return False


def calculate_performance_stats(workflows_df, matching_workflows, metrics):
    """
    Calculate performance statistics for workflows matching a rule.
    
    Args:
        workflows_df: Full workflows DataFrame
        matching_workflows: Boolean mask of workflows matching the rule
        metrics: List of metric column names
    
    Returns:
        Dictionary with statistics for each metric
    """
    stats = {}
    
    if matching_workflows.sum() == 0:
        return stats
    
    for metric in metrics:
        if metric not in workflows_df.columns:
            continue
            
        values = workflows_df.loc[matching_workflows, metric]
        
        if len(values) < 2:
            stats[metric] = {
                'mean': values.iloc[0] if len(values) == 1 else np.nan,
                'std': 0.0,
                'cv': 0.0,
                'min': values.iloc[0] if len(values) == 1 else np.nan,
                'max': values.iloc[0] if len(values) == 1 else np.nan,
                'q25': values.iloc[0] if len(values) == 1 else np.nan,
                'q50': values.iloc[0] if len(values) == 1 else np.nan,
                'q75': values.iloc[0] if len(values) == 1 else np.nan,
                'n_samples': len(values),
                'consistency': 'PERFECT' if len(values) == 1 else 'N/A'
            }
        else:
            mean_val = values.mean()
            std_val = values.std()
            cv = abs(std_val / mean_val) if mean_val != 0 else np.inf
            
            # Consistency rating based on coefficient of variation
            if cv < 0.1:
                consistency = 'EXCELLENT'
            elif cv < 0.25:
                consistency = 'GOOD'
            elif cv < 0.5:
                consistency = 'MODERATE'
            else:
                consistency = 'POOR'
            
            stats[metric] = {
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': values.min(),
                'max': values.max(),
                'q25': values.quantile(0.25),
                'q50': values.quantile(0.50),
                'q75': values.quantile(0.75),
                'n_samples': len(values),
                'consistency': consistency
            }
    
    return stats


def validate_rule(workflows_df, rule_row, metrics, cluster_id):
    """
    Validate a single rule by checking consistency and performance.
    
    Args:
        workflows_df: Full workflows DataFrame
        rule_row: Row from cluster_decision_rules.csv
        metrics: List of metric column names
        cluster_id: Expected cluster ID
    
    Returns:
        Dictionary with validation results
    """
    rule_str = rule_row['rule']
    
    # Find workflows matching this rule
    matching_mask = workflows_df.apply(lambda row: parse_rule_condition(rule_str, row), axis=1)
    n_matching = matching_mask.sum()
    
    # Find workflows in the target cluster
    in_cluster = workflows_df['cluster'] == cluster_id
    
    # True positives: matching rule AND in cluster
    true_positives = (matching_mask & in_cluster).sum()
    
    # False positives: matching rule but NOT in cluster
    false_positives = (matching_mask & ~in_cluster).sum()
    
    # Calculate actual precision (may differ from reported due to data changes)
    actual_precision = true_positives / n_matching if n_matching > 0 else 0
    
    # Get performance stats for matching workflows
    performance_stats = calculate_performance_stats(workflows_df, matching_mask, metrics)
    
    # Validation checks
    validation = {
        'rule': rule_str,
        'n_matching': n_matching,
        'n_in_cluster': true_positives,
        'n_outside_cluster': false_positives,
        'actual_precision': actual_precision,
        'reported_precision': rule_row['precision'],
        'precision_match': abs(actual_precision - rule_row['precision']) < 0.01,
        'performance_stats': performance_stats
    }
    
    return validation


def generate_recommendations(validation_results, metrics_of_interest):
    """
    Generate recommendations for which rule to use based on desired outcomes.
    
    Args:
        validation_results: List of validation result dictionaries
        metrics_of_interest: List of metric names to consider
    
    Returns:
        Dictionary of recommendations
    """
    recommendations = {}
    
    for metric in metrics_of_interest:
        metric_rules = []
        
        for result in validation_results:
            if metric not in result['performance_stats']:
                continue
            
            stats = result['performance_stats'][metric]
            
            if stats['n_samples'] < 2:
                continue
            
            metric_rules.append({
                'cluster': result['cluster_id'],
                'rule_number': result['rule_number'],
                'rule': result['rule'],
                'mean': stats['mean'],
                'std': stats['std'],
                'cv': stats['cv'],
                'consistency': stats['consistency'],
                'n_workflows': stats['n_samples'],
                'precision': result['actual_precision']
            })
        
        if not metric_rules:
            continue
        
        # Sort by mean (descending for "high" recommendations)
        sorted_high = sorted(metric_rules, key=lambda x: x['mean'], reverse=True)
        sorted_low = sorted(metric_rules, key=lambda x: x['mean'], reverse=False)
        
        # Filter for good consistency and decent sample size
        reliable_rules = [r for r in metric_rules if r['consistency'] in ['EXCELLENT', 'GOOD'] and r['n_workflows'] >= 3]
        
        recommendations[metric] = {
            'highest_mean': sorted_high[0] if sorted_high else None,
            'lowest_mean': sorted_low[0] if sorted_low else None,
            'most_reliable_high': sorted([r for r in reliable_rules], key=lambda x: x['mean'], reverse=True)[0] if reliable_rules else None,
            'most_consistent': sorted(metric_rules, key=lambda x: x['cv'])[0] if metric_rules else None
        }
    
    return recommendations


def save_validation_results(validation_results, recommendations, output_dir):
    """Save validation results to JSON and CSV files."""
    
    # Save detailed validation results
    validation_json = []
    for result in validation_results:
        # Convert performance_stats to serializable format
        perf_stats_serializable = {}
        for metric, stats in result['performance_stats'].items():
            perf_stats_serializable[metric] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                for k, v in stats.items()
            }
        
        validation_json.append({
            'cluster_id': int(result['cluster_id']),
            'rule_number': int(result['rule_number']),
            'rule': result['rule'],
            'n_matching': int(result['n_matching']),
            'n_in_cluster': int(result['n_in_cluster']),
            'n_outside_cluster': int(result['n_outside_cluster']),
            'actual_precision': float(result['actual_precision']),
            'reported_precision': float(result['reported_precision']),
            'precision_match': bool(result['precision_match']),
            'performance_stats': perf_stats_serializable
        })
    
    with open(output_dir / 'rule_validation_results.json', 'w') as f:
        json.dump(validation_json, f, indent=2)
    
    # Save recommendations
    recs_serializable = {}
    for metric, recs in recommendations.items():
        recs_serializable[metric] = {}
        for rec_type, rec_data in recs.items():
            if rec_data is not None:
                recs_serializable[metric][rec_type] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in rec_data.items()
                }
            else:
                recs_serializable[metric][rec_type] = None
    
    with open(output_dir / 'rule_recommendations.json', 'w') as f:
        json.dump(recs_serializable, f, indent=2)
    
    # Create CSV summary
    summary_rows = []
    for result in validation_results:
        row = {
            'cluster_id': result['cluster_id'],
            'rule_number': result['rule_number'],
            'rule': result['rule'],
            'n_matching': result['n_matching'],
            'actual_precision': result['actual_precision'],
            'reported_precision': result['reported_precision'],
            'precision_validated': result['precision_match']
        }
        
        # Add key metrics
        for metric in ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC-ROC']:
            if metric in result['performance_stats']:
                stats = result['performance_stats'][metric]
                row[f'{metric}_mean'] = stats['mean']
                row[f'{metric}_std'] = stats['std']
                row[f'{metric}_consistency'] = stats['consistency']
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'rule_validation_summary.csv', index=False)
    
    print(f"\n✓ Validation results saved to: {output_dir / 'rule_validation_results.json'}")
    print(f"✓ Recommendations saved to: {output_dir / 'rule_recommendations.json'}")
    print(f"✓ Summary CSV saved to: {output_dir / 'rule_validation_summary.csv'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_validate_cluster_rules.py data/workflows")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Load data
    print("Loading data...")
    workflows_df = pd.read_csv(data_dir / 'workflows_clustered.csv')
    
    # Normalize column names (replace spaces with underscores)
    workflows_df.columns = workflows_df.columns.str.replace(' ', '_')
    
    rules_df = pd.read_csv(data_dir / 'cluster_decision_rules.csv')
    
    # Load metric names
    with open(data_dir / 'metric_names.txt', 'r') as f:
        metric_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(workflows_df)} workflows")
    print(f"Loaded {len(rules_df)} rules across {rules_df['cluster_id'].nunique()} clusters")
    print()
    
    # Validate each rule
    validation_results = []
    
    for idx, rule_row in rules_df.iterrows():
        print("rulerow")
        print(rule_row)
        cluster_id = rule_row['cluster_id']
        rule_number = rule_row['rule_number']
        
        print(f"Validating Cluster {cluster_id} - Rule {rule_number}...")
        
        validation = validate_rule(workflows_df, rule_row, metric_names, cluster_id)
        validation['cluster_id'] = cluster_id
        validation['rule_number'] = rule_number
        
        validation_results.append(validation)
    
    # Print validation report
    
    # Generate and print recommendations
    key_metrics = [
        'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC',
        'learning_curve_train_score', 'Test Fairness'
    ]
    
    recommendations = generate_recommendations(validation_results, key_metrics)
    
    # Save results
    save_validation_results(validation_results, recommendations, data_dir)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
