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
    'xlabel': 26,
    'ylabel': 26,
    'xtick': 26,
    'ytick': 26,
    'legend': 24,
    'figure': 12,
    'figsize': [8, 6]
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
    
    SeparationErr(E_c) = |{x ∈ X | E_c(x) = True ∧ CL(x) ∈ C\{c}}| / |{x ∈ X | E(x) = true}|
    
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
    
    Conciseness(E_c) = 1 / |{P | P is a predicate in E_c}|
    
    Counts the actual number of predicates (conditions) in the rule.
    """
    n_predicates = count_predicates_in_rule(rule_str)
    if n_predicates == 0:
        return 0.0
    return 1.0 / n_predicates


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
    fig, ax = plt.subplots()
    
    clusters = qse_df['cluster_id'].values
    x = np.arange(len(clusters))
    width = 0.25
    
    # Plot the three components
    bars1 = ax.bar(x - width, qse_df['coverage'], width, 
                   label='Coverage', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, qse_df['separation_quality'], width,
                   label='Separation Quality (1-Error)', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, qse_df['conciseness'], width,
                   label='Conciseness', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE components plot saved to {save_path}")


def plot_qse_scores(qse_df: pd.DataFrame, save_path: str = None):
    """
    Plot overall QSE scores for each cluster.
    """
    fig, ax = plt.subplots()
    
    clusters = qse_df['cluster_id'].values
    qse_scores = qse_df['qse'].values
    
    bars = ax.bar(clusters, qse_scores, color='#9b59b6', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10)
    
    # Add horizontal line for average QSE
    avg_qse = qse_scores.mean()
    ax.axhline(y=avg_qse, color='red', linestyle='--', linewidth=2, 
               label=f'Average QSE: {avg_qse:.3f}')
    
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Quality Score for Explanation (QSE)')
    ax.set_xticks(clusters)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE scores plot saved to {save_path}")


def plot_qse_per_cluster_all_rules(qse_all_rules_df: pd.DataFrame, save_path: str = None):
    """
    Plot QSE scores for all rules within each cluster.
    Creates a grouped bar chart showing how QSE varies across rules for each cluster.
    """
    clusters = sorted(qse_all_rules_df['cluster_id'].unique())
    n_clusters = len(clusters)
    
    # Determine grid layout
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(clusters):
        ax = axes[idx]
        cluster_data = qse_all_rules_df[qse_all_rules_df['cluster_id'] == cluster_id]
        
        rules = cluster_data['rule_number'].values
        x = np.arange(len(rules))
        width = 0.2
        
        # Plot components
        ax.bar(x - 1.5*width, cluster_data['coverage'], width, 
               label='Coverage', color='#3498db', alpha=0.8, edgecolor='black')
        ax.bar(x - 0.5*width, cluster_data['separation_quality'], width,
               label='Sep. Quality', color='#2ecc71', alpha=0.8, edgecolor='black')
        ax.bar(x + 0.5*width, cluster_data['conciseness'], width,
               label='Conciseness', color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.bar(x + 1.5*width, cluster_data['qse'], width,
               label='QSE', color='#9b59b6', alpha=0.8, edgecolor='black')
        
        # Formatting
        ax.set_xlabel('Rule Number')
        ax.set_ylabel('Score')
        ax.set_title(f'Cluster {cluster_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(rules)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE per cluster (all rules) plot saved to {save_path}")


def plot_qse_heatmap_all_rules(qse_all_rules_df: pd.DataFrame, save_path: str = None):
    """
    Plot a heatmap showing QSE scores for all cluster-rule combinations.
    """
    # Pivot data to create a matrix: clusters x rules
    pivot_data = qse_all_rules_df.pivot(index='cluster_id', columns='rule_number', values='qse')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    
    # Labels
    ax.set_xlabel('Rule Number')
    ax.set_ylabel('Cluster ID')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('QSE Score', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not pd.isna(value):
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE heatmap saved to {save_path}")



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
    print(cluster_metrics_df.head(3))

    return cluster_metrics_df

def plot_silhouette_analysis(X_scaled, clusters, silhouette_avg, per_cluster_metrics_df, save_path=None):
    fig, ax1 = plt.subplots(1, 1)
    # fig.set_size_inches(10, 7)

    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(X_scaled) + (len(per_cluster_metrics_df['cluster_id']) + 1) * 10])
    
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)
    y_lower = 10

    for i in sorted(per_cluster_metrics_df['cluster_id']):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(per_cluster_metrics_df['cluster_id']))
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    # ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.2, 1.1, 0.2))
    
    # plt.suptitle(f"Silhouette analysis for clustering with n_clusters = {len(per_cluster_metrics_df['cluster_id'])}",
    #              fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Silhouette plot saved to {save_path}")
    # plt.show()

def plot_silhouette_boxplot(X_scaled, clusters, silhouette_avg, per_cluster_metrics_df, save_path=None):
    """
    Generates a box plot of silhouette scores for each cluster.
    """
    fig, ax = plt.subplots()
    
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)
    
    cluster_ids = sorted(per_cluster_metrics_df['cluster_id'].unique())
    data_to_plot = [sample_silhouette_values[clusters == i] for i in cluster_ids]
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=[str(i) for i in cluster_ids])
    
    # Customizing colors
    colors = [plt.cm.nipy_spectral(float(i) / len(cluster_ids)) for i in cluster_ids]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.axhline(y=silhouette_avg, color="red", linestyle="--", label=f'Overall Avg Silhouette: {silhouette_avg:.2f}')
    
    # ax.set_title('Distribution of Silhouette Scores per Cluster', fontsize=16)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylim([-0.2, 1.0])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Silhouette box plot saved to {save_path}")
    # plt.show()

def plot_combined_cluster_quality(scores_df: pd.DataFrame, save_path: str = None):
    metrics = scores_df.columns
    values = scores_df.iloc[0].values
    
    fig, axes = plt.subplots(1, len(metrics))
    # fig.suptitle('Overall Cluster Quality Scores', fontsize=16)
    
    colors = ['#4c72b0', '#55a868', '#c44e52']
    
    for i, (metric, value, color) in enumerate(zip(metrics, values, colors)):
        axes[i].bar(metric, value, color=color, edgecolor='white')
        # axes[i].set_title(metric.replace('_', ' '), fontsize=12)
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text label for the value
        axes[i].text(0, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Combined quality plot saved to {save_path}")
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

def plot_predictive_quality(predictive_df: pd.DataFrame, save_path: str = None):
    plt.figure()
    x = np.arange(len(predictive_df))
    width = 0.2  # Adjusted width to fit 4 bars comfortably

    # Centering the bars: -1.5, -0.5, +0.5, +1.5 times width
    plt.bar(x - 1.5*width, predictive_df['F1 Score'], width, label='F1 Score', color='#4c72b0', edgecolor='white')
    plt.bar(x - 0.5*width, predictive_df['Balanced Accuracy'], width, label='Balanced Accuracy', color='#55a868', edgecolor='white')
    plt.bar(x + 0.5*width, predictive_df['ROC AUC'], width, label='ROC AUC', color='#c44e52', edgecolor='white')
    plt.bar(x + 1.5*width, predictive_df['Quality Score'], width, label='Quality Score', color='#333333', edgecolor='white')

    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    # plt.title('Predictive quality', fontsize=14)
    plt.xticks(x, predictive_df['Cluster'], fontsize=10)
    plt.ylim(0, 1.15)  # Extra space for labels if needed
    plt.legend(loc='upper center', ncol=4, frameon=True)  # Legend below chart
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    # plt.show()

def plot_predictive_quality_table(predictive_df: pd.DataFrame, save_path: str = None):
    """
    Plot predictive quality metrics as a formatted table.
    """
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table - transpose so clusters are columns
    metrics = ['F1 Score', 'Balanced Accuracy', 'ROC AUC', 'Quality Score']
    clusters = predictive_df['Cluster'].tolist()
    
    # Create table data with metrics as rows and clusters as columns
    table_data = []
    for metric in metrics:
        row = [f"{val:.4f}" for val in predictive_df[metric].values]
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data, 
                     rowLabels=metrics,
                     colLabels=clusters,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * len(clusters))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Color header row (cluster labels)
    for i in range(len(clusters)):
        table[(0, i)].set_facecolor('#4c72b0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color metric rows alternating
    colors = ['#f0f0f0', '#e0e0e0']
    for i in range(len(metrics)):
        for j in range(len(clusters)):
            table[(i+1, j)].set_facecolor(colors[i % 2])
    
    # Color row labels (metrics)
    for i in range(len(metrics)):
        table[(i+1, -1)].set_facecolor('#2d5986')
        table[(i+1, -1)].set_text_props(weight='bold', color='white')
    
    # plt.title('Predictive Quality Metrics by Cluster', pad=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Predictive quality table saved to {save_path}")
    # plt.show()

####################################RULE  QUALITY####################################
def convert_rule_to_pandas(rule_str):
    # Replace "IN {...}" with "in [...]"
    # The regex captures the content inside {} and puts it inside []
    rule_str = re.sub(r"IN\s*\{([^}]+)\}", r"in [\1]", rule_str)
    
    # Replace single = with == for equality checks
    # We look for = that is NOT preceded by <, >, or ! and NOT followed by =
    rule_str = re.sub(r"(?<![<>!])=(?!=)", "==", rule_str)
    
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
            
        selected_features_raw = clusters_insights[cluster_key]['feature_selection']['selected_features']
        
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

def plot_rule_quality(cv_summary_df: pd.DataFrame, save_path: str = None):
    
    # Get unique clusters and rules
    clusters = sorted(cv_summary_df['Cluster'].unique())
    rules = sorted(cv_summary_df['Rule'].unique())

    n_clusters = len(clusters)
    n_rules = len(rules)

    # Setup plot
    fig, ax = plt.subplots()

    # Width of each bar
    bar_width = 0.8 / n_rules

    # X locations for the groups
    indices = np.arange(n_clusters)

    for i, rule in enumerate(rules):
        # Extract data for this rule
        rule_data = cv_summary_df[cv_summary_df['Rule'] == rule]
        
        # Align data with clusters (fill missing with NaN/0)
        medians = []
        lower_errs = []
        upper_errs = []
        
        for cluster in clusters:
            match = rule_data[rule_data['Cluster'] == cluster]
            if not match.empty:
                median = match.iloc[0]['Median_CV']
                p10 = match.iloc[0]['CV_10th_Percentile']
                p90 = match.iloc[0]['CV_90th_Percentile']
                
                medians.append(median)
                # Error bars are relative to the top of the bar (median)
                lower_errs.append(median - p10)
                upper_errs.append(p90 - median)
            else:
                medians.append(0)
                lower_errs.append(0)
                upper_errs.append(0)
                
        # Calculate x positions for this rule's bars
        x_positions = indices + (i - (n_rules - 1) / 2) * bar_width
        
        # Plot bars with error bars
        # The bar height is the Median CV
        # The error bars extend from the 10th percentile to the 90th percentile
        ax.bar(x_positions, medians, bar_width, 
            yerr=[lower_errs, upper_errs], 
            capsize=4, 
            label=f'Rule {rule}',
            alpha=0.8,
            edgecolor='black')

    # Formatting
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    # ax.set_title('Stability of Discriminative Metrics within Rules\n(Bar = Median CV, Whiskers = 10th-90th Percentile Range)', fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(clusters)
    ax.legend(title='Rule Number')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Rule quality plot saved to {save_path}")

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
    ax.set_xticklabels([f'Cluster {c}' for c in clusters])
    
    # Create legend for rules
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=rule_colors[i], edgecolor='black', 
                             label=f'Rule {rule}', alpha=0.7) 
                      for i, rule in enumerate(rules)]
    ax.legend(handles=legend_elements, title='Rule Number', loc='best')
    
    # Formatting
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
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
    fig, ax = plt.subplots(figsize=(14, 6))
    
    clusters = df['Cluster'].values
    x = np.arange(len(clusters))
    width = 0.35
    
    # Plot bars for selected and non-selected metrics
    bars1 = ax.bar(x - width/2, df['Selected_CV_Mean'], width, 
                   label='Selected Metrics', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Non_Selected_CV_Mean'], width,
                   label='Non-Selected Metrics', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Mean Coefficient of Variation (CV)')
    # ax.set_title('Stability Comparison: Selected vs Non-Selected Metrics\n(Lower CV = More Stable)')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Representative metrics CV comparison plot saved to {save_path}")

def representative_metrics_quality(clustered_data: pd.DataFrame, clusters_insights: dict, metric_cols: list) -> pd.DataFrame:
    """
    Compare CV distribution for selected (discriminative) vs non-selected metrics.
    
    For each cluster:
    - Selected metrics = those in feature_selection.selected_features
    - Non-selected metrics = all other metrics
    
    We expect selected metrics to have lower CV (more stable within cluster).
    
    Returns:
        DataFrame with CV statistics for selected vs non-selected metrics per cluster
    """
    results = []
    
    for cluster_id_str, cluster_info in clusters_insights.items():
        cluster_id = int(cluster_id_str)
        
        # Get selected features for this cluster
        selected_features_raw = cluster_info['feature_selection']['selected_features']
        
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



    clustered_data, clusters_insights, X_scaled, metric_cols = read_data(path)

    ##Cluster Quality 
    db_index, ch_score, sil_score, all_scores = calculate_cluster_quality(clustered_data, X_scaled)
    cluster_silhouette_df = silhouette_statistics(clustered_data, X_scaled)
    os.makedirs(result_dir_cluster_quality, exist_ok=True)
    all_scores.to_csv(os.path.join(result_dir_cluster_quality, "overall_cluster_quality_scores.csv"), index=False)
    cluster_silhouette_df.to_csv(os.path.join(result_dir_cluster_quality, "per_cluster_silhouette_metrics.csv"), index=False)
    plot_silhouette_analysis(X_scaled, clustered_data['cluster'], sil_score, cluster_silhouette_df, save_path=os.path.join(result_dir_cluster_quality, "silhouette_plot.png"))
    plot_silhouette_boxplot(X_scaled, clustered_data['cluster'], sil_score, cluster_silhouette_df, save_path=os.path.join(result_dir_cluster_quality, "silhouette_boxplot.png"))
    plot_combined_cluster_quality(all_scores, save_path=os.path.join(result_dir_cluster_quality, "combined_quality_scores.png"))

    ##Predictive Quality 
    predictive_df = calculate_predictive_quality(clusters_insights)
    os.makedirs(result_dir_predictive_quality, exist_ok=True)
    if not predictive_df.empty and len(predictive_df) > 0:
        predictive_df.to_csv(os.path.join(result_dir_predictive_quality, "predictive_quality_metrics.csv"), index=False)
        plot_predictive_quality(predictive_df, save_path=os.path.join(result_dir_predictive_quality, "predictive_quality.png"))
        plot_predictive_quality_table(predictive_df, save_path=os.path.join(result_dir_predictive_quality, "predictive_quality_table.png"))
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
        plot_rule_quality(cv_summary_df, save_path=os.path.join(result_dir_rule_quality, "rule_quality.png"))
        plot_rule_quality_box_plot(cv_summary_df, save_path=os.path.join(result_dir_rule_quality, "rule_quality_boxplot.png"))
    else:
        print("Warning: No decision tree rules found. Skipping rule quality analysis.")
   
    ## Representative Metrics Quality
    os.makedirs(result_dir_representative_quality, exist_ok=True)
    representative_quality_df = representative_metrics_quality(clustered_data, clusters_insights, metric_cols)
    representative_quality_df.to_csv(os.path.join(result_dir_representative_quality, "representative_quality_metrics.csv"), index=False)
    plot_representative_metrics_cv_comparison(representative_quality_df, save_path=os.path.join(result_dir_representative_quality, "cv_comparison.png"))
    
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
        plot_qse_scores(qse_best_df, save_path=os.path.join(result_dir_qse, "qse_scores.png"))
        
        # Plot QSE for all rules
        plot_qse_per_cluster_all_rules(qse_all_rules_df, save_path=os.path.join(result_dir_qse, "qse_per_cluster_all_rules.png"))
        plot_qse_heatmap_all_rules(qse_all_rules_df, save_path=os.path.join(result_dir_qse, "qse_heatmap.png"))
    else:
        print("Warning: No rules available. Skipping QSE calculation.")



