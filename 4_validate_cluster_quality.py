import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_samples

FONT_SIZE = {
    'title': 22,
    'xlabel': 40,
    'ylabel': 40,
    'xtick': 32,
    'ytick': 32,
    'legend': 25,
    'figure': 12,
    'figsize': [10, 8]
}
 
# Set publication-ready plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'Palatino', 'DejaVu Serif'],
    'font.size': FONT_SIZE['figure'],
    'axes.titlesize': FONT_SIZE['title'],
    'axes.labelsize': FONT_SIZE['xlabel'],
    'xtick.labelsize': FONT_SIZE['xtick'],
    'ytick.labelsize': FONT_SIZE['ytick'],
    'legend.fontsize': FONT_SIZE['legend'],
    'figure.figsize': FONT_SIZE['figsize'],
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.labelpad': 10
})


def read_data(file_path: str) -> pd.DataFrame:
    clustered_data = pd.read_csv(file_path+"/workflows_clustered.csv")
    clustered_data=clustered_data.drop(columns=["workflowId"])

    with open(file_path+"/clusters_comprehensive_insights.json") as f:
        clusters_insights = json.load(f)

    metrics_file = os.path.join(file_path, 'metric_names.txt')
    with open(metrics_file, 'r') as f:
        metric_cols = [line.strip() for line in f.readlines() if line.strip()]

    available_cols = set(clustered_data.columns)
    metric_cols = [col for col in metric_cols if col in available_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustered_data[metric_cols].values)

    return clustered_data, clusters_insights, X_scaled, metric_cols

def read_processed_pcs(file_path: str):
    """
    Load the processed data used to create clusters (principal components)
    and prepare scaled features for cluster quality validation.

    Expects a CSV with columns: 'PC_*' and 'cluster'.
    Automatically detects and uses all available principal component columns
    matching the pattern 'PC_*' (e.g., PC_1, PC_2, PC_3, ...).
    """
    pcs_path = os.path.join(file_path, 'workflows_processed_data.csv')
    if not os.path.exists(pcs_path):
        raise FileNotFoundError(f"Processed PCs file not found: {pcs_path}")

    pcs_df = pd.read_csv(pcs_path)
    
    # Detect all principal component columns automatically
    pc_cols = [c for c in pcs_df.columns if re.match(r'^PC_\d+$', str(c))]
    
    if 'cluster' not in pcs_df.columns:
        raise ValueError("Processed PCs file missing required column: 'cluster'")
    if not pc_cols:
        raise ValueError("Processed PCs file contains no principal component columns (expected columns like 'PC_1', 'PC_2', ...)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pcs_df[pc_cols].values)

    return pcs_df, X_scaled


####################################EXPLANATION QUALITY SCORE (QSE)####################################

def count_predicates_in_rule(rule_str: str) -> int:
    """
    Count the number of predicates in a rule string.
    Predicates are separated by 'and' operators.
    """
    if not rule_str or pd.isna(rule_str):
        return 0
    # Split by 'and' and count non-empty parts
    predicates = [p.strip() for p in rule_str.split(' and ') if p.strip()]
    return len(predicates)


def calculate_rule_based_coverage(rule_str: str, cluster_id: int, all_data: pd.DataFrame) -> float:
    """
    Calculate coverage based on actual decision rule.
    
    Coverage(E_c) = |{x ∈ X | E_c(x) = true ∧ CL(x) = c}| / |{x ∈ X | CL(x) = c}|
    
    This is the ratio of points in cluster c that satisfy the rule.
    Note: This is equivalent to the 'recall' metric already calculated for rules.
    """
    if not rule_str or pd.isna(rule_str):
        return 0.0
    
    # Convert rule to pandas query format
    pandas_rule = convert_rule_to_pandas(rule_str)
    
    try:
        # Get all points in the target cluster
        cluster_data = all_data[all_data['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        if cluster_size == 0:
            return 0.0
        
        # Apply rule to get points that satisfy it
        satisfying_data = cluster_data.query(pandas_rule)
        
        # Coverage = points in cluster that satisfy rule / total points in cluster
        coverage = len(satisfying_data) / cluster_size
        return coverage
        
    except Exception as e:
        print(f"Error evaluating rule for cluster {cluster_id}: {e}")
        return 0.0


def calculate_rule_based_separation_error(rule_str: str, cluster_id: int, all_data: pd.DataFrame) -> float:
    """
    Calculate separation error based on actual decision rule.
    
    SeparationErr(E_c) = |{x ∈ X | E_c(x) = True ∧ CL(x) ∈ C\\{c}}| / |{x ∈ X | E(x) = true}|
    
    This is the ratio of points satisfying the rule that DON'T belong to cluster c.
    Note: This is equivalent to (1 - precision) of the rule.
    """
    if not rule_str or pd.isna(rule_str):
        return 1.0
    
    # Convert rule to pandas query format
    pandas_rule = convert_rule_to_pandas(rule_str)
    
    try:
        # Apply rule to all data to get points that satisfy it
        satisfying_data = all_data.query(pandas_rule)
        total_satisfying = len(satisfying_data)
        
        if total_satisfying == 0:
            return 0.0
        
        # Count points that satisfy the rule but are NOT in cluster c
        other_cluster_satisfying = len(satisfying_data[satisfying_data['cluster'] != cluster_id])
        
        # Separation error = wrong cluster points / all satisfying points
        separation_error = other_cluster_satisfying / total_satisfying
        return separation_error
        
    except Exception as e:
        print(f"Error evaluating rule for cluster {cluster_id}: {e}")
        return 1.0


def calculate_rule_based_conciseness(rule_str: str) -> float:
    """
    Calculate conciseness based on actual decision rule.
    
    Conciseness(E_c) decays gently with predicate count so longer rules are penalized
    without collapsing to near-zero too quickly.
    
    Counts the actual number of predicates (conditions) in the rule.
    """
    n_predicates = count_predicates_in_rule(rule_str)
    if n_predicates == 0:
        return 0.0
    decay = 0.25  # smaller decay -> slower drop in conciseness
    return 1.0 / (1.0 + decay * (n_predicates - 1))


def calculate_qse_for_rule(rule_str: str, cluster_id: int, all_data: pd.DataFrame) -> dict:
    """
    Calculate Quality Score for Explanation (QSE) for a single rule.
    
    Based on the paper's Section 3.2, QSE combines three quality measures:
    1. Coverage: ratio of points in cluster c for which explanation (rule) holds
    2. Separation Error: ratio of non-cluster points that satisfy the explanation
    3. Conciseness: inverse of number of predicates in the rule
    
    QSE(E_c) = [Coverage(E_c) + (1 - SeparationErr(E_c)) + Conciseness(E_c)] / 3
    
    Args:
        rule_str: The decision rule string
        cluster_id: ID of the cluster
        all_data: DataFrame with all workflows (must include 'cluster' column)
        
    Returns:
        Dictionary with coverage, separation error, conciseness, and QSE scores
    """
    # Calculate the three quality measures based on the actual rule
    coverage = calculate_rule_based_coverage(rule_str, cluster_id, all_data)
    separation_err = calculate_rule_based_separation_error(rule_str, cluster_id, all_data)
    conciseness = calculate_rule_based_conciseness(rule_str)
    
    # Calculate QSE as the average of the three normalized measures
    # Note: Separation error is inverted (1 - err) to align with "higher is better"
    qse = (coverage + (1 - separation_err) + conciseness) / 3
    
    n_predicates = count_predicates_in_rule(rule_str)
    
    return {
        'coverage': coverage,
        'separation_error': separation_err,
        'separation_quality': 1 - separation_err,
        'conciseness': conciseness,
        'qse': qse,
        'n_predicates': n_predicates
    }


def calculate_all_qse_from_rules(rules_df: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate QSE for all cluster rules.
    
    Args:
        rules_df: DataFrame with cluster decision rules
        all_data: DataFrame with all workflows (must include 'cluster' column)
    
    Returns:
        DataFrame with QSE metrics for each cluster-rule combination
    """
    results = []
    
    for _, row in rules_df.iterrows():
        cluster_id = row['cluster_id']
        rule_num = row['rule_number']
        rule_str = row['rule']
        
        # Calculate QSE for this rule
        qse_metrics = calculate_qse_for_rule(rule_str, cluster_id, all_data)
        
        # Combine with rule info
        result = {
            'cluster_id': cluster_id,
            'rule_number': rule_num,
            **qse_metrics
        }
        results.append(result)
    
    qse_df = pd.DataFrame(results)
    qse_df = qse_df.sort_values(['cluster_id', 'rule_number'])
    
    return qse_df


def calculate_best_qse_per_cluster(qse_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, select the rule with the highest QSE score.
    
    Returns:
        DataFrame with best QSE metrics per cluster
    """
    # Group by cluster and get the row with max QSE
    best_qse = qse_df.loc[qse_df.groupby('cluster_id')['qse'].idxmax()]
    return best_qse.reset_index(drop=True)


def plot_qse_components(qse_df: pd.DataFrame, save_path: str = None):
    """
    Plot QSE components (Coverage, Separation Quality, Conciseness) for each cluster.
    """
    fig, ax = plt.subplots(figsize=tuple(FONT_SIZE['figsize']))
    
    clusters = qse_df['cluster_id'].values
    x = np.arange(len(clusters))
    width = 0.25
    
    # Plot the three components
    bars1 = ax.bar(x - width, qse_df['coverage'], width, 
                   label='Coverage', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, qse_df['separation_quality'], width,
                   label='Separation Quality (1-Error)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, qse_df['conciseness'], width,
                   label='Conciseness', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=FONT_SIZE['figure'], fontweight='bold')
    
    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('Score', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, fontsize=FONT_SIZE['xtick'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    ax.legend(fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE components plot saved to {save_path}")


def plot_qse_scores(qse_df: pd.DataFrame, save_path: str = None):
    """
    Plot overall QSE scores for each cluster.
    """
    fig, ax = plt.subplots(figsize=tuple(FONT_SIZE['figsize']))
    
    clusters = qse_df['cluster_id'].values
    qse_scores = qse_df['qse'].values
    
    bars = ax.bar(clusters, qse_scores, color='#4292C6', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=FONT_SIZE['figure'], fontweight='bold')
    
    # Add horizontal line for average QSE
    avg_qse = qse_scores.mean()
    ax.axhline(y=avg_qse, color='black', linestyle='--', linewidth=2.5, 
               label=f'Average QSE: {avg_qse:.3f}')
    
    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('QSE', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.set_xticks(clusters)
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE scores plot saved to {save_path}")


####################################CLUSTER QUALITY##################################
def calculate_cluster_quality(clustered_data: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    scaled_davies_bouldin_score = davies_bouldin_score(
    X_scaled, 
    clustered_data['cluster'].values
)
    print(f"Davies-Bouldin Index: {scaled_davies_bouldin_score:.4f}")
    print(f"  Range: [0, ∞) | Lower is better | Excellent: <0.5, Good: <1.0, Fair: <1.5")
    print()
    scaled_calinski_harabasz_score = calinski_harabasz_score(
        X_scaled, 
        clustered_data['cluster'].values
    )
    print(f"Calinski-Harabasz Score: {scaled_calinski_harabasz_score:.2f}")
    print(f"  Range: [0, ∞) | Higher is better | No fixed threshold (depends on data)")
    print(f"  → Measures ratio of between-cluster to within-cluster dispersion")
    print(f"  → Higher values indicate denser, better-separated clusters")

    print()

    scaled_silhouette_score = silhouette_score(
        X_scaled, 
        clustered_data['cluster'].values
    )
    print(f"Silhouette Score: {scaled_silhouette_score:.4f}")
    print(f"  Range: [-1, 1] | Higher is better | Excellent: >0.7, Good: >0.5, Fair: >0.25")

    all_df= pd.DataFrame({
        'Davies_Bouldin_Index': [scaled_davies_bouldin_score],
        'Calinski_Harabasz_Score': [scaled_calinski_harabasz_score],
        'Silhouette_Score': [scaled_silhouette_score]
    })


    return scaled_davies_bouldin_score, scaled_calinski_harabasz_score, scaled_silhouette_score, all_df



def plot_representative_metrics_cv_boxplot(detailed_df: pd.DataFrame, save_path: str = None):
    """
    Create box plots comparing CV distribution for selected vs non-selected metrics per cluster.
    
    Args:
        detailed_df: DataFrame with individual CV values per metric (from representative_metrics_quality_detailed)
        save_path: Path to save the plot
    """
    from matplotlib.patches import Patch
    
    # Create REAL box plots with actual data
    clusters = sorted(detailed_df['Cluster'].unique())

    all_data = []
    positions = []

    pos = 1
    for cluster in clusters:
        cluster_data = detailed_df[detailed_df['Cluster'] == cluster]
        
        # Get actual CV values (not simulated!)
        selected_cvs = cluster_data[cluster_data['Type'] == 'Selected']['CV'].values
        non_selected_cvs = cluster_data[cluster_data['Type'] == 'Non-Selected']['CV'].values
        
        all_data.append(selected_cvs)
        all_data.append(non_selected_cvs)
        positions.append(pos)
        positions.append(pos + 0.5)
        pos += 1.5

    # Create box plot with explicit figure size to match bar chart
    fig, ax = plt.subplots(figsize=tuple(FONT_SIZE['figsize']))

    bp = ax.boxplot(all_data, 
                    positions=positions,
                    patch_artist=True,
                    showmeans=False,
                    widths=0.4)

    # Colorblind-friendly colors: blue for good (representative), orange for bad (other)
    # These colors are distinguishable for all types of colorblindness and in grayscale
    if ablation=="no_iterative_filter":
        color_bad = '#DE8F05'       # Vermillion - other metrics (bad)
        color_good = '#DE8F05'      # Green - representative metrics (good)
    else:
        color_good = '#0173B2' 
            # Blue - representative metrics (good)
        color_bad = '#DE8F05'       # Orange - other metrics (bad)

    colors = [color_good, color_bad] * len(clusters)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    # Set x-axis to show only cluster numbers
    cluster_positions = [pos + 0.25 for pos in positions[::2]]
    ax.set_xticks(cluster_positions)
    ax.set_xticklabels([f'{c}' for c in clusters], fontsize=FONT_SIZE['xtick'])

    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('CV', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    # ax.set_title('Selected vs Non-Selected CV Distribution by Cluster - Adult Dataset (Real Data)', 
    #              fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add legend with colorblind-friendly colors
    from matplotlib.patches import Patch
    if ablation=="no_iterative_filter":
        legend_elements = [Patch(facecolor=color_good, edgecolor='black', linewidth=1.2, label='All Metrics'),
                    ]
    else:
        legend_elements = [Patch(facecolor=color_good, edgecolor='black', linewidth=1.2, label='Representative Metrics'),
                    Patch(facecolor=color_bad, edgecolor='black', linewidth=1.2, label='Other Metrics')]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path)
        print(f"Representative metrics CV boxplot saved to {save_path}")
    
    return fig, ax

def representative_metrics_quality_detailed(clustered_data: pd.DataFrame, clusters_insights: dict, metric_cols: list):
    """
    Returns both summary statistics AND individual CV values for box plots.
    """
    summary_results = []
    detailed_cvs = []  # Store individual CV values
    
    for cluster_id_str, cluster_info in clusters_insights.items():
        cluster_id = int(cluster_id_str)
        
        # Get selected features for this cluster
        cluster_data = cluster_info
        distinct_features = cluster_data.get('distinct_features', {}).get('features', [])
        
        # Handle different structures of distinct_features
        if isinstance(distinct_features, dict) and 'features' in distinct_features:
            selected_features_raw = distinct_features['features']
        elif isinstance(distinct_features, list) and len(distinct_features) > 0:
            selected_features_raw = distinct_features
        else:
            # Fallback to selected features from feature_selection step
            selected_features_raw = cluster_data.get('feature_selection', {}).get('selected_features', [])
        
        selected_features = selected_features_raw
        # Filter to only include metrics that exist in our metric_cols
        selected_metrics = [f for f in selected_features if f in metric_cols]
        non_selected_metrics = [f for f in metric_cols if f not in selected_metrics]
        
        # Get data for this cluster only
        cluster_mask = clustered_data['cluster'] == cluster_id
        cluster_data = clustered_data[cluster_mask]
        
        if len(cluster_data) < 2:
            continue
        
        # Calculate CV for selected metrics
        selected_cvs = []
        for metric in selected_metrics:
            if metric in cluster_data.columns:
                mean_val = cluster_data[metric].mean()
                std_val = cluster_data[metric].std()
                
                if abs(mean_val) > 1e-9:
                    cv = std_val / abs(mean_val)
                    selected_cvs.append(cv)
                    # Store individual CV value
                    detailed_cvs.append({
                        'Cluster': cluster_id,
                        'Metric': metric,
                        'Type': 'Selected',
                        'CV': cv
                    })
        
        # Calculate CV for non-selected metrics
        non_selected_cvs = []
        for metric in non_selected_metrics:
            if metric in cluster_data.columns:
                mean_val = cluster_data[metric].mean()
                std_val = cluster_data[metric].std()
                
                if abs(mean_val) > 1e-9:
                    cv = std_val / abs(mean_val)
                    non_selected_cvs.append(cv)
                    # Store individual CV value
                    detailed_cvs.append({
                        'Cluster': cluster_id,
                        'Metric': metric,
                        'Type': 'Non-Selected',
                        'CV': cv
                    })
        
        # Compute summary statistics
        summary_results.append({
            'Cluster': cluster_id,
            'N_Workflows': len(cluster_data),
            'N_Selected_Metrics': len(selected_metrics),
            'N_Non_Selected_Metrics': len(non_selected_metrics),
            'Selected_CV_Mean': np.mean(selected_cvs) if selected_cvs else np.nan,
            'Selected_CV_Median': np.median(selected_cvs) if selected_cvs else np.nan,
            'Selected_CV_Std': np.std(selected_cvs) if selected_cvs else np.nan,
            'Non_Selected_CV_Mean': np.mean(non_selected_cvs) if non_selected_cvs else np.nan,
            'Non_Selected_CV_Median': np.median(non_selected_cvs) if non_selected_cvs else np.nan,
            'Non_Selected_CV_Std': np.std(non_selected_cvs) if non_selected_cvs else np.nan,
            'CV_Difference': (np.mean(non_selected_cvs) - np.mean(selected_cvs)) if (selected_cvs and non_selected_cvs) else np.nan
        })
    
    summary_df = pd.DataFrame(summary_results)
    detailed_df = pd.DataFrame(detailed_cvs)
    
    return summary_df, detailed_df


def silhouette_statistics(clustered_data: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    silhouette_vals = silhouette_samples(X_scaled, clustered_data['cluster'].values)
    unique_labels = np.unique(clustered_data['cluster'].values)
    cluster_metrics = []
        
    for cluster_id in unique_labels:
        cluster_mask = clustered_data['cluster'].values == cluster_id
        cluster_silhouette_vals = silhouette_vals[cluster_mask]
        
        cluster_metrics.append({
            'cluster_id': cluster_id,
            'n_samples': np.sum(cluster_mask),
            'silhouette_avg': np.mean(cluster_silhouette_vals),
            'silhouette_median': np.median(cluster_silhouette_vals),
            'silhouette_std': np.std(cluster_silhouette_vals),
            'silhouette_min': np.min(cluster_silhouette_vals),
            'silhouette_max': np.max(cluster_silhouette_vals)
        })
            
        
        
    cluster_metrics_df = pd.DataFrame(cluster_metrics)
    print("\nPer-Cluster Silhouette Metrics:")
    print(cluster_metrics_df)

    return cluster_metrics_df

def plot_silhouette_boxplot(X_scaled, clusters, silhouette_avg, per_cluster_metrics_df, save_path=None):
    """
    Generates a box plot of silhouette scores for each cluster.
    """
    fig, ax = plt.subplots(figsize=tuple(FONT_SIZE['figsize']))
    
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)
    
    cluster_ids = sorted(per_cluster_metrics_df['cluster_id'].unique())
    data_to_plot = [sample_silhouette_values[clusters == i] for i in cluster_ids]
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=[str(i) for i in cluster_ids], widths=0.5)
    
    # Customizing colors and edges
    colors = [plt.cm.nipy_spectral(float(i) / len(cluster_ids)) for i in cluster_ids]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style whiskers and other elements
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.2, linestyle='--', alpha=0.7)
    for cap in bp['caps']:
        cap.set(linewidth=1.2)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)
        
    ax.axhline(y=silhouette_avg, color="black", linestyle='--', linewidth=2.5, label=f'Overall Avg Silhouette: {silhouette_avg:.2f}')
    
    # ax.set_title('Distribution of Silhouette Scores per Cluster', fontsize=16)
    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('Silhouette Coefficient', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.tick_params(axis='x', labelsize=FONT_SIZE['xtick'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    ax.set_ylim([-0.2, 1.0])
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Silhouette box plot saved to {save_path}")
    # plt.show()

####################################PREDICTIVE QUALITY####################################

def calculate_predictive_quality(clusters_insights: dict) -> pd.DataFrame:
    predictive_metrics = []
    for cluster_id, data in clusters_insights.items():
        if 'model_evaluation' in data and data['model_evaluation']:
            eval_data = data['model_evaluation']
            predictive_metrics.append({
                'Cluster': f"Cluster {cluster_id}",
                'F1 Score': eval_data.get('f1_score', 0),
                'Balanced Accuracy': eval_data.get('balanced_accuracy', 0),
                'ROC AUC': eval_data.get('test_auc', 0),
                'Quality Score': eval_data.get('model_quality_score', 0),
                'Interpretation': eval_data.get('quality_interpretation', 'N/A')
            })

    predictive_df = pd.DataFrame(predictive_metrics)
    return predictive_df

####################################RULE  QUALITY####################################
def convert_rule_to_pandas(rule_str):
    """
    Convert decision tree rule format to pandas query format.
    
    Handles:
    - col = None -> col.isna()
    - col = 'value' -> col == 'value'
    - col IN {None, 'val1', 'val2'} -> (col.isna() | col.in(['val1', 'val2']))
    - col IN {'val1', 'val2'} -> col.in(['val1', 'val2'])
    - and -> &
    """
    import re
    
    # First handle "IN {...}" patterns that contain None
    def replace_in_with_none(match):
        col = match.group(1)
        values_str = match.group(2)
        
        # Check if None is in the list
        if 'None' in values_str:
            # Split values and filter out None, keeping quoted strings
            values = [v.strip() for v in values_str.split(',')]
            non_none_values = [v for v in values if v != 'None']
            
            if non_none_values:
                # Has both None and other values
                values_list = ', '.join(non_none_values)
                return f"({col}.isna() | {col}.isin([{values_list}]))"
            else:
                # Only None
                return f"{col}.isna()"
        else:
            # No None, just normal IN list
            return f"{col}.isin([{values_str}])"
    
    # Replace "col IN {...}" patterns
    rule_str = re.sub(r'(\w+)\s+IN\s*\{([^}]+)\}', replace_in_with_none, rule_str)
    
    # Handle "col = None" -> "col.isna()"
    rule_str = re.sub(r'(\w+)\s*=\s*None\b', r'\1.isna()', rule_str)
    
    # Replace single = with == for equality checks (but not for <=, >=, !=)
    rule_str = re.sub(r"(?<![<>!])=(?!=)", "==", rule_str)
    
    # Replace " and " with " & " for pandas query
    rule_str = re.sub(r'\s+and\s+', ' & ', rule_str)
    
    return rule_str

def overall_for_rules(rules_df: pd.DataFrame, raw_data: pd.DataFrame, cluster_labels: pd.DataFrame): 
    analysis_data = pd.merge(raw_data, cluster_labels, on='workflowId')
    analysis_data.columns = [c.replace(' ', '_') for c in analysis_data.columns]
    sub_frames = {}
    results = []
    for index, row in rules_df.iterrows():
        cluster_id = row['cluster_id']
        rule_num = row['rule_number']
        rule_raw = row['rule']
        
        pandas_rule = convert_rule_to_pandas(rule_raw)
        
        try:
            segment = analysis_data.query(pandas_rule)
            sub_frames[(cluster_id, rule_num)] = segment
            support = len(segment) / len(analysis_data)
            target_in_segment = len(segment[segment['cluster'] == cluster_id])
            precision = target_in_segment / len(segment) if len(segment) > 0 else 0
            total_target = len(analysis_data[analysis_data['cluster'] == cluster_id])
            recall = target_in_segment / total_target if total_target > 0 else 0
            
            results.append({
                'cluster_id': cluster_id,
                'rule_number': rule_num,
                'rule': rule_raw,
                'segment_size': len(segment),
                'support': support,
                'precision': precision,
                'recall': recall
            })
            
        except Exception as e:
            print(f"Error processing rule for Cluster {cluster_id}, Rule {rule_num}: {e}")

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)
    return results_df, sub_frames

    # Display the first few rows of the results
    # print(f"Created {len(sub_frames)} sub-frames.")
    # results_df[['cluster_id', 'rule_number', 'segment_size', 'support', 'precision', 'recall']].head(21)
    ##RULE  QUALITY##

def rule_quality(sub_frames: dict, clusters_insights: dict):
    # Initialize list to store CV summary results
    cv_results = []

    print("Calculating Coefficient of Variation (CV) for discriminative metrics within rule segments...")

    for (cluster_id, rule_num), segment in sub_frames.items():
        # 1. Get selected features (discriminative metrics) for this cluster
        # The JSON keys are strings "0", "1", etc., so we convert cluster_id to string
        cluster_key = str(cluster_id)
        if cluster_key not in clusters_insights:
            print(f"Warning: No insights found for Cluster {cluster_id}")
            continue
        
        # Try to get SHAP features first (if available and populated), otherwise use selected features
        cluster_data = clusters_insights[cluster_key]
        distinct_features = cluster_data.get('distinct_features', {}).get('features', [])
        
        # Handle different structures of distinct_features
        if isinstance(distinct_features, dict) and 'features' in distinct_features:
            selected_features_raw = distinct_features['features']
        elif isinstance(distinct_features, list) and len(distinct_features) > 0:
            selected_features_raw = distinct_features
        else:
            # Fallback to selected features from feature_selection step
            selected_features_raw = cluster_data.get('feature_selection', {}).get('selected_features', [])
        
        # 2. Map feature names to dataframe columns (spaces to underscores)
        selected_features = [f.replace(' ', '_') for f in selected_features_raw]
        
        # Filter to keep only features that exist in our data
        valid_features = [f for f in selected_features if f in segment.columns]
        
        if not valid_features:
            print(f"Warning: No valid features found for Cluster {cluster_id} (Rule {rule_num})")
            continue
            
        # 3. Compute CV for each feature in this segment
        # CV = std / mean
        # We use absolute value of mean to handle negative values (though raw metrics are usually positive)
        # We add a small epsilon to avoid division by zero
        
        feature_cvs = []
        
        for feature in valid_features:
            feat_data = segment[feature]
            
            # Skip if constant (std=0) or empty
            if len(feat_data) < 2:
                continue
                
            mean_val = feat_data.mean()
            std_val = feat_data.std()
            
            if abs(mean_val) < 1e-9:
                # If mean is effectively zero, CV is undefined/infinite. 
                # We might skip it or assign a high value. Here we skip.
                continue
                
            cv = std_val / abs(mean_val)
            feature_cvs.append(cv)
        
        if not feature_cvs:
            continue
            
        # 4. Summarize CVs for this rule
        cv_median = np.median(feature_cvs)
        cv_10th = np.percentile(feature_cvs, 10)
        cv_90th = np.percentile(feature_cvs, 90)
        
        cv_results.append({
            'Cluster': cluster_id,
            'Rule': rule_num,
            'Num_Features': len(feature_cvs),
            'Median_CV': cv_median,
            'CV_10th_Percentile': cv_10th,
            'CV_90th_Percentile': cv_90th
        })

    # Create the summary DataFrame
    cv_summary_df = pd.DataFrame(cv_results)

    # Sort for better readability
    cv_summary_df = cv_summary_df.sort_values(['Cluster', 'Rule'])

    # Display the dataframe
    return cv_summary_df

def plot_rule_quality_box_plot(cv_summary_df: pd.DataFrame, save_path: str = None):
    """
    Creates boxplots showing the distribution of CV values for each rule within each cluster.
    Similar structure to plot_rule_quality but using box plots instead of bars.
    """
    # Get unique clusters and rules
    clusters = sorted(cv_summary_df['Cluster'].unique())
    rules = sorted(cv_summary_df['Rule'].unique())
    
    n_clusters = len(clusters)
    n_rules = len(rules)
    
    # Setup plot
    fig, ax = plt.subplots()
    
    # Prepare data for grouped boxplots
    # Each rule will get its own set of boxes across clusters
    box_positions = []
    data_to_plot = []
    colors_list = []
    
    # Generate distinct colors for each rule
    rule_colors = [plt.cm.Set3(i / n_rules) for i in range(n_rules)]
    
    # Width for spacing
    cluster_spacing = 1.0
    box_width = 0.6 / n_rules
    
    for i, rule in enumerate(rules):
        for j, cluster in enumerate(clusters):
            # Find data for this cluster-rule combination
            match = cv_summary_df[(cv_summary_df['Cluster'] == cluster) & (cv_summary_df['Rule'] == rule)]
            
            if not match.empty:
                row = match.iloc[0]
                # Create distribution from percentiles
                cv_values = [row['CV_10th_Percentile'], row['Median_CV'], row['CV_90th_Percentile']]
                data_to_plot.append(cv_values)
                
                # Calculate position: cluster_position + rule_offset
                pos = j * cluster_spacing + (i - (n_rules - 1) / 2) * box_width
                box_positions.append(pos)
                colors_list.append(rule_colors[i])
            else:
                # No data for this combination - skip
                pass
    
    # Create boxplot with custom positions
    bp = ax.boxplot(data_to_plot, positions=box_positions, widths=box_width * 0.8, 
                     patch_artist=True, showfliers=False)
    
    # Color the boxes by rule
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    # Customize whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.2, linestyle='--', alpha=0.6)
    for cap in bp['caps']:
        cap.set(linewidth=1.2)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)
    
    # Set x-axis labels at cluster centers
    cluster_positions = [i * cluster_spacing for i in range(n_clusters)]
    ax.set_xticks(cluster_positions)
    ax.set_xticklabels([f'{c}' for c in clusters], fontsize=FONT_SIZE['xtick'])
    
    # Create legend for rules
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=rule_colors[i], edgecolor='black', 
                             label=f'Rule {rule}', alpha=0.7, linewidth=1.5) 
                      for i, rule in enumerate(rules)]
    ax.legend(handles=legend_elements, title='Rule Number', loc='best', 
              fontsize=FONT_SIZE['legend'], title_fontsize=FONT_SIZE['legend'], 
              frameon=True, edgecolor='black', fancybox=False)
    
    # Formatting
    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('Coefficient of Variation (CV)', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    # ax.set_title('Stability of Discriminative Metrics within Rules\n(Box plots showing CV distribution per rule)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Rule quality box plot saved to {save_path}")

####################################REPRESENTATIVE METRICS QUALITY####################################

def plot_representative_metrics_cv_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Create a simple bar plot comparing CV for selected vs non-selected metrics per cluster.
    """
    fig, ax = plt.subplots(figsize=tuple(FONT_SIZE['figsize']))
    
    clusters = df['Cluster'].values
    x = np.arange(len(clusters))
    width = 0.35
    
    # Plot bars for selected and non-selected metrics
    bars1 = ax.bar(x - width/2, df['Selected_CV_Mean'], width, 
                   label='Selected Metrics', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, df['Non_Selected_CV_Mean'], width,
                   label='Non-Selected Metrics', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=FONT_SIZE['figure'], fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Cluster', fontweight='bold', fontsize=FONT_SIZE['xlabel'])
    ax.set_ylabel('Mean Coefficient of Variation (CV)', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    # ax.set_title('Stability Comparison: Selected vs Non-Selected Metrics\n(Lower CV = More Stable)')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, fontsize=FONT_SIZE['xtick'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    ax.legend(fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Representative metrics CV comparison plot saved to {save_path}")

def representative_metrics_quality(clustered_data: pd.DataFrame, clusters_insights: dict, metric_cols: list) -> pd.DataFrame:
    """
    Compare CV distribution for selected (discriminative) vs non-selected metrics.
    
    For each cluster:
    - Selected metrics = those in distinct_features.features
    - Non-selected metrics = all other metrics
    
    We expect selected metrics to have lower CV (more stable within cluster).
    
    Returns:
        DataFrame with CV statistics for selected vs non-selected metrics per cluster
    """
    results = []
    
    for cluster_id_str, cluster_info in clusters_insights.items():
        cluster_id = int(cluster_id_str)
        
        # Get selected features for this cluster
        cluster_data = cluster_info
        distinct_features = cluster_data.get('distinct_features', {}).get('features', [])
        
        # Handle different structures of distinct_features
        if isinstance(distinct_features, dict) and 'features' in distinct_features:
            selected_features_raw = distinct_features['features']
        elif isinstance(distinct_features, list) and len(distinct_features) > 0:
            selected_features_raw = distinct_features
        else:
            # Fallback to selected features from feature_selection step
            selected_features_raw = cluster_data.get('feature_selection', {}).get('selected_features', [])
        
        # Don't normalize - the column names in the dataframe have spaces, matching the JSON
        selected_features = selected_features_raw
        
        # Filter to only include metrics that exist in our metric_cols
        selected_metrics = [f for f in selected_features if f in metric_cols]
        non_selected_metrics = [f for f in metric_cols if f not in selected_metrics]
        
        # Get data for this cluster only
        cluster_mask = clustered_data['cluster'] == cluster_id
        cluster_data = clustered_data[cluster_mask]
        
        if len(cluster_data) < 2:
            print(f"Skipping Cluster {cluster_id}: insufficient data")
            continue
        
        # Calculate CV for selected metrics
        selected_cvs = []
        for metric in selected_metrics:
            if metric in cluster_data.columns:
                mean_val = cluster_data[metric].mean()
                std_val = cluster_data[metric].std()
                
                if abs(mean_val) > 1e-9:  # Avoid division by zero
                    cv = std_val / abs(mean_val)
                    selected_cvs.append(cv)
        
        # Calculate CV for non-selected metrics
        non_selected_cvs = []
        for metric in non_selected_metrics:
            if metric in cluster_data.columns:
                mean_val = cluster_data[metric].mean()
                std_val = cluster_data[metric].std()
                
                if abs(mean_val) > 1e-9:  # Avoid division by zero
                    cv = std_val / abs(mean_val)
                    non_selected_cvs.append(cv)
        
        # Compute summary statistics
        results.append({
            'Cluster': cluster_id,
            'N_Workflows': len(cluster_data),
            'N_Selected_Metrics': len(selected_metrics),  # Use actual selected metrics count
            'N_Non_Selected_Metrics': len(non_selected_metrics),  # Use actual non-selected metrics count
            'Selected_CV_Mean': np.mean(selected_cvs) if selected_cvs else np.nan,
            'Selected_CV_Median': np.median(selected_cvs) if selected_cvs else np.nan,
            'Selected_CV_Std': np.std(selected_cvs) if selected_cvs else np.nan,
            'Non_Selected_CV_Mean': np.mean(non_selected_cvs) if non_selected_cvs else np.nan,
            'Non_Selected_CV_Median': np.median(non_selected_cvs) if non_selected_cvs else np.nan,
            'Non_Selected_CV_Std': np.std(non_selected_cvs) if non_selected_cvs else np.nan,
            'CV_Difference': (np.mean(non_selected_cvs) - np.mean(selected_cvs)) if (selected_cvs and non_selected_cvs) else np.nan
        })
    
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    import sys
    
    # Default values
    # path = "./data/workflows"
    # dataset_name = "adult"
    # ablation = "full"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]
    if len(sys.argv) > 3:
        ablation = sys.argv[3]
    
    print(f"Configuration:")
    print(f"  Data path: {path}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Ablation: {ablation}")
    print()
    
    result_dir_cluster_quality = f"./results/{dataset_name}/{ablation}/cluster_quality"
    result_dir_predictive_quality = f"./results/{dataset_name}/{ablation}/predictive_quality"
    result_dir_rule_quality = f"./results/{dataset_name}/{ablation}/rule_quality"
    result_dir_representative_quality = f"./results/{dataset_name}/{ablation}/representative_quality"
    result_dir_qse = f"./results/{dataset_name}/{ablation}/explanation_quality"

    # Load clustered workflows for analyses that need raw metrics
    clustered_data, clusters_insights, X_scaled_metrics, metric_cols = read_data(path)
    # Load processed PCs specifically for cluster-quality validation

    if ablation != "no_dim_reduction":
        pcs_df, X_scaled_pcs = read_processed_pcs(path)


        db_index, ch_score, sil_score, all_scores = calculate_cluster_quality(pcs_df, X_scaled_pcs)
        cluster_silhouette_df = silhouette_statistics(pcs_df, X_scaled_pcs)
        os.makedirs(result_dir_cluster_quality, exist_ok=True)
        all_scores.to_csv(os.path.join(result_dir_cluster_quality, "overall_cluster_quality_scores.csv"), index=False)
        cluster_silhouette_df.to_csv(os.path.join(result_dir_cluster_quality, "per_cluster_silhouette_metrics.csv"), index=False)
        plot_silhouette_boxplot(X_scaled_pcs, pcs_df['cluster'], sil_score, cluster_silhouette_df, save_path=os.path.join(result_dir_cluster_quality, "silhouette_boxplot.png"))
    else:
        db_index, ch_score, sil_score, all_scores = calculate_cluster_quality(clustered_data, X_scaled_metrics)
        cluster_silhouette_df = silhouette_statistics(clustered_data, X_scaled_metrics)
        os.makedirs(result_dir_cluster_quality, exist_ok=True)
        all_scores.to_csv(os.path.join(result_dir_cluster_quality, "overall_cluster_quality_scores.csv"), index=False)
        cluster_silhouette_df.to_csv(os.path.join(result_dir_cluster_quality, "per_cluster_silhouette_metrics.csv"), index=False)
        plot_silhouette_boxplot(X_scaled_metrics, clustered_data['cluster'], sil_score, cluster_silhouette_df, save_path=os.path.join(result_dir_cluster_quality, "silhouette_boxplot.png"))
    ##Predictive Quality 
    predictive_df = calculate_predictive_quality(clusters_insights)
    os.makedirs(result_dir_predictive_quality, exist_ok=True)
    if not predictive_df.empty and len(predictive_df) > 0:
        predictive_df.to_csv(os.path.join(result_dir_predictive_quality, "predictive_quality_metrics.csv"), index=False)
    else:
        print("Warning: No predictive quality metrics available. Skipping predictive quality plots.")
       
    ##Rule quality
    rules_list = []
    for cluster_id_str, cluster_info in clusters_insights.items():
        cluster_id = int(cluster_id_str)
        if 'decision_tree_rules' in cluster_info and cluster_info['decision_tree_rules']:
            for rule_num, rule_data in enumerate(cluster_info['decision_tree_rules'], start=1):
                rules_list.append({
                    'cluster_id': cluster_id,
                    'rule_number': rule_num,
                    'rule': rule_data['rule'],
                    'precision': rule_data.get('precision', 0),
                    'recall': rule_data.get('recall', 0),
                    'f1_score': rule_data.get('f1_score', 0),
                    'n_workflows_in_cluster': rule_data.get('n_workflows_in_cluster', 0),
                    'combined_score': rule_data.get('combined_score', 0)
                })
    
    if rules_list:
        rules_df = pd.DataFrame(rules_list)
        raw_data_full = pd.read_csv(f"{path}/workflows.csv")
        cluster_labels = pd.read_csv(f"{path}/workflows_clustered.csv")[['workflowId', 'cluster']]
        _,sub_frames=overall_for_rules(rules_df, raw_data_full, cluster_labels)
        cv_summary_df=rule_quality(sub_frames, clusters_insights)
        os.makedirs(result_dir_rule_quality, exist_ok=True)
        cv_summary_df.to_csv(os.path.join(result_dir_rule_quality, "rule_quality_metrics.csv"), index=False)
        plot_rule_quality_box_plot(cv_summary_df, save_path=os.path.join(result_dir_rule_quality, "rule_quality_boxplot.png"))
    else:
        print("Warning: No decision tree rules found. Skipping rule quality analysis.")
   
    ## Representative Metrics Quality
    os.makedirs(result_dir_representative_quality, exist_ok=True)
    summary_df, detailed_df = representative_metrics_quality_detailed(clustered_data, clusters_insights, metric_cols)
    detailed_df.to_csv(os.path.join(result_dir_representative_quality, "representative_quality_detailed.csv"), index=False)
    plot_representative_metrics_cv_boxplot(detailed_df, save_path=os.path.join(result_dir_representative_quality, "cluster_respresentative_quality.png"))

    # representative_quality_df = representative_metrics_quality(clustered_data, clusters_insights, metric_cols)
    # representative_quality_df.to_csv(os.path.join(result_dir_representative_quality, "representative_quality_metrics.csv"), index=False)
    # plot_representative_metrics_cv_comparison(representative_quality_df, save_path=os.path.join(result_dir_representative_quality, "cv_comparison.png"))
    
    ## Explanation Quality Score (QSE)
    os.makedirs(result_dir_qse, exist_ok=True)
    
    if rules_list:
        # Load workflow data for QSE calculation
        raw_data_full = pd.read_csv(f"{path}/workflows.csv")
        cluster_labels = pd.read_csv(f"{path}/workflows_clustered.csv")[['workflowId', 'cluster']]
        all_data_with_clusters = pd.merge(raw_data_full, cluster_labels, on='workflowId')
        all_data_with_clusters.columns = [c.replace(' ', '_') for c in all_data_with_clusters.columns]
        
        # Calculate QSE for all rules
        qse_all_rules_df = calculate_all_qse_from_rules(rules_df, all_data_with_clusters)
        qse_all_rules_df.to_csv(os.path.join(result_dir_qse, "qse_all_rules.csv"), index=False)
        
        # Get best QSE per cluster
        qse_best_df = calculate_best_qse_per_cluster(qse_all_rules_df)
        qse_best_df.to_csv(os.path.join(result_dir_qse, "qse_best_per_cluster.csv"), index=False)
        
        # Plot QSE for best rules per cluster
        plot_qse_components(qse_best_df, save_path=os.path.join(result_dir_qse, "qse_components.png"))
        plot_qse_scores(qse_best_df, save_path=os.path.join(result_dir_qse, f"explanation_quality_qse.png"))
    else:
        print("Warning: No rules available. Skipping QSE calculation.")


def plot_datasets_scores(ablation="full", results_base_path="./results", output_dir="./paper_results"):
    """Plot cluster quality metrics across all datasets for a given ablation mode."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cluster_wine = pd.read_csv(f"{results_base_path}/wine/{ablation}/cluster_quality/overall_cluster_quality_scores.csv")
        cluster_adult = pd.read_csv(f"{results_base_path}/adult/{ablation}/cluster_quality/overall_cluster_quality_scores.csv")
        cluster_taxi = pd.read_csv(f"{results_base_path}/taxi/{ablation}/cluster_quality/overall_cluster_quality_scores.csv")

        cluster_wine = cluster_wine.drop(columns=["Calinski_Harabasz_Score"])
        cluster_adult = cluster_adult.drop(columns=["Calinski_Harabasz_Score"])
        cluster_taxi = cluster_taxi.drop(columns=["Calinski_Harabasz_Score"])

        metrics = ['DBI', 'SiS']
        wine_scores = [
            cluster_wine['Davies_Bouldin_Index'].values[0],
            cluster_wine['Silhouette_Score'].values[0]
        ]
        adult_scores = [
            cluster_adult['Davies_Bouldin_Index'].values[0],
            cluster_adult['Silhouette_Score'].values[0]
        ]
        taxi_scores = [
            cluster_taxi['Davies_Bouldin_Index'].values[0],
            cluster_taxi['Silhouette_Score'].values[0]
        ]

        # Create bar plot with colorblind-friendly blue palette
        x = np.arange(len(metrics))
        width = 0.15

        # Colorblind-friendly blue palette (light to dark blue)
        colors = {
            'wine': '#9ECAE1',    # Light blue
            'adult': '#4292C6',   # Medium blue  
            'taxi': '#08519C'     # Dark blue
        }

        # Hatching patterns for black & white printing distinction
        hatches = {
            'wine': '',       # No hatch (solid)
            'adult': '//',    # Diagonal lines
            'taxi': 'xx'      # Cross-hatch
        }

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width, wine_scores, width, label='Wine', 
                    color=colors['wine'], edgecolor='black', linewidth=1.2,
                    hatch=hatches['wine'])
        bars2 = ax.bar(x, adult_scores, width, label='Adult',
                    color=colors['adult'], edgecolor='black', linewidth=1.2,
                    hatch=hatches['adult'])
        bars3 = ax.bar(x + width, taxi_scores, width, label='Taxi',
                    color=colors['taxi'], edgecolor='black', linewidth=1.2,
                    hatch=hatches['taxi'])

        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_quality_metrics_{ablation}.png")
        plt.close()
        
        print(f"✓ Plotted cluster quality metrics for ablation: {ablation}")
    except Exception as e:
        print(f"Warning: Could not plot cluster quality metrics for ablation '{ablation}': {e}")


def copy_quality_images(results_base_path="./results", output_dir="./paper_results"):
    """Copy representative and explanation quality images to paper_results directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Copy adult/full representative quality
        img = plt.imread(f"{results_base_path}/adult/full/representative_quality/cluster_respresentative_quality.png")
        plt.imsave(f"{output_dir}/cluster_respresentative_quality_full_adult.png", img)
        
        # Copy adult/full explanation quality
        img = plt.imread(f"{results_base_path}/adult/full/explanation_quality/explanation_quality_qse.png")
        plt.imsave(f"{output_dir}/explanation_quality_qse_adult_full.png", img)
        
        # Copy adult/no_iterative_filter representative quality
        img = plt.imread(f"{results_base_path}/adult/no_iterative_filter/representative_quality/cluster_respresentative_quality.png")
        plt.imsave(f"{output_dir}/cluster_respresentative_quality_no_iterative_filter_adult.png", img)

         # Copy taxi/no_variance_filter representative quality
        img = plt.imread(f"{results_base_path}/taxi/no_variance_filter/representative_quality/cluster_respresentative_quality.png")
        plt.imsave(f"{output_dir}/cluster_respresentative_quality_no_variance_filter_taxi.png", img)
        
          # Copy wine/no_variance_filter representative quality
        img = plt.imread(f"{results_base_path}/wine/no_variance_filter/representative_quality/cluster_respresentative_quality.png")
        plt.imsave(f"{output_dir}/cluster_respresentative_quality_no_variance_filter_wine.png", img)
        
         # Copy wine/full representative quality
        img = plt.imread(f"{results_base_path}/wine/full/representative_quality/cluster_respresentative_quality.png")
        plt.imsave(f"{output_dir}/cluster_respresentative_quality_full_wine.png", img)
        
        print(f"✓ Copied quality images to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not copy quality images: {e}")
