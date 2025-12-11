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

def calculate_coverage(rule_str: str, df: pd.DataFrame, cluster_id: int) -> float:
    """
    Calculate coverage for a rule explanation.
    Coverage = |{x ∈ X | E_c(x) = true ∧ CL(x) = c}| / |{x ∈ X | CL(x) = c}|
    
    Args:
        rule_str: The rule string to evaluate
        df: DataFrame with all data and cluster labels
        cluster_id: The target cluster ID
    
    Returns:
        Coverage ratio (0 to 1)
    """
    # Total points in cluster c
    total_in_cluster = len(df[df['cluster'] == cluster_id])
    
    if total_in_cluster == 0:
        return 0.0
    
    # Points in cluster c where rule holds
    pandas_rule = convert_rule_to_pandas(rule_str)
    try:
        rule_satisfies = df.query(pandas_rule)
        rule_satisfies_in_cluster = len(rule_satisfies[rule_satisfies['cluster'] == cluster_id])
        coverage = rule_satisfies_in_cluster / total_in_cluster
        return coverage
    except Exception as e:
        print(f"Error evaluating rule for cluster {cluster_id}: {e}")
        return 0.0

def calculate_separation_error(rule_str: str, df: pd.DataFrame, cluster_id: int) -> float:
    """
    Calculate separation error for a rule explanation.
    SeparationErr = |{x ∈ X | E_c(x) = True ∧ CL(x) ∈ C \ {c}}| / |{x ∈ X | E(x) = true}|
    
    Args:
        rule_str: The rule string to evaluate
        df: DataFrame with all data and cluster labels
        cluster_id: The target cluster ID
    
    Returns:
        Separation error ratio (0 to 1)
    """
    pandas_rule = convert_rule_to_pandas(rule_str)
    try:
        # All points where rule holds
        rule_satisfies = df.query(pandas_rule)
        total_rule_satisfies = len(rule_satisfies)
        
        if total_rule_satisfies == 0:
            return 0.0
        
        # Points where rule holds but NOT in cluster c
        rule_satisfies_other_clusters = len(rule_satisfies[rule_satisfies['cluster'] != cluster_id])
        separation_error = rule_satisfies_other_clusters / total_rule_satisfies
        return separation_error
    except Exception as e:
        print(f"Error evaluating rule for cluster {cluster_id}: {e}")
        return 1.0  # Worst case

def calculate_conciseness(rule_str: str) -> float:
    """
    Calculate conciseness for a rule explanation.
    Conciseness = 1 / |{P | P is a predicate in E_c}|
    
    Args:
        rule_str: The rule string to evaluate
    
    Returns:
        Conciseness score (0 to 1)
    """
    # Count predicates by splitting on 'and' and 'or' (case insensitive)
    # Rules are typically connected with 'and'
    predicates = re.split(r'\s+and\s+|\s+or\s+', rule_str, flags=re.IGNORECASE)
    num_predicates = len(predicates)
    
    if num_predicates == 0:
        return 0.0
    
    conciseness = 1.0 / num_predicates
    return conciseness

def calculate_qse(coverage: float, separation_error: float, conciseness: float) -> float:
    """
    Calculate Quality Score for Explanation (QSE).
    QSE = (Coverage + (1 - SeparationErr) + Conciseness) / 3
    
    Args:
        coverage: Coverage score (0 to 1)
        separation_error: Separation error (0 to 1)
        conciseness: Conciseness score (0 to 1)
    
    Returns:
        QSE score (0 to 1)
    """
    qse = (coverage + (1 - separation_error) + conciseness) / 3.0
    return qse

def calculate_all_qse_metrics(rules_df: pd.DataFrame, df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all QSE metrics for all rules.
    
    Args:
        rules_df: DataFrame with columns ['cluster_id', 'rule_number', 'rule', ...]
        df_with_clusters: DataFrame with all data including 'cluster' column
    
    Returns:
        DataFrame with QSE metrics for each rule
    """
    qse_results = []
    
    for index, row in rules_df.iterrows():
        cluster_id = row['cluster_id']
        rule_num = row['rule_number']
        rule_str = row['rule']
        
        # Calculate metrics
        coverage = calculate_coverage(rule_str, df_with_clusters, cluster_id)
        separation_error = calculate_separation_error(rule_str, df_with_clusters, cluster_id)
        conciseness = calculate_conciseness(rule_str)
        qse = calculate_qse(coverage, separation_error, conciseness)
        
        qse_results.append({
            'cluster_id': cluster_id,
            'rule_number': rule_num,
            'rule': rule_str,
            'coverage': coverage,
            'separation_error': separation_error,
            'conciseness': conciseness,
            'qse': qse,
            'num_predicates': int(1 / conciseness) if conciseness > 0 else 0
        })
    
    qse_df = pd.DataFrame(qse_results)
    return qse_df

def plot_qse_metrics(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a grouped bar plot showing Coverage, 1-SeparationErr, and Conciseness for each rule.
    """
    # Get unique clusters
    clusters = sorted(qse_df['cluster_id'].unique())
    
    fig, axes = plt.subplots(1, len(clusters), squeeze=False)
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(clusters):
        ax = axes[idx]
        cluster_data = qse_df[qse_df['cluster_id'] == cluster_id].sort_values('rule_number')
        
        rules = cluster_data['rule_number'].values
        x = np.arange(len(rules))
        width = 0.25
        
        # Plot the three components of QSE
        ax.bar(x - width, cluster_data['coverage'].values, width, 
               label='Coverage', color='#3498db', alpha=0.8, edgecolor='black')
        ax.bar(x, 1 - cluster_data['separation_error'].values, width, 
               label='1 - Sep. Error', color='#2ecc71', alpha=0.8, edgecolor='black')
        ax.bar(x + width, cluster_data['conciseness'].values, width, 
               label='Conciseness', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Rule Number')
        ax.set_ylabel('Score')
        ax.set_title(f'Cluster {cluster_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(rules)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE components plot saved to {save_path}")

def plot_qse_overall(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a bar plot showing overall QSE score for each rule.
    """
    # Get unique clusters
    clusters = sorted(qse_df['cluster_id'].unique())
    
    fig, ax = plt.subplots()
    
    # Prepare data for grouped bars
    cluster_positions = {}
    bar_width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, qse_df['rule_number'].max()))
    
    # Calculate positions for each cluster
    for i, cluster_id in enumerate(clusters):
        cluster_positions[cluster_id] = i
    
    # Plot bars for each rule number across clusters
    max_rules = qse_df.groupby('cluster_id')['rule_number'].max().max()
    
    for rule_num in range(1, int(max_rules) + 1):
        rule_data = qse_df[qse_df['rule_number'] == rule_num]
        positions = []
        qse_values = []
        
        for cluster_id in clusters:
            cluster_rule = rule_data[rule_data['cluster_id'] == cluster_id]
            if not cluster_rule.empty:
                positions.append(cluster_positions[cluster_id] + (rule_num - 1) * bar_width)
                qse_values.append(cluster_rule['qse'].values[0])
        
        ax.bar(positions, qse_values, bar_width, 
               label=f'Rule {rule_num}', color=colors[rule_num - 1], 
               alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('QSE Score')
    ax.set_title('Quality Score for Explanations (QSE) by Cluster and Rule')
    ax.set_xticks([cluster_positions[c] + bar_width * (max_rules - 1) / 2 for c in clusters])
    ax.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE overall plot saved to {save_path}")

def plot_qse_heatmap(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a heatmap showing QSE scores for all cluster-rule combinations.
    """
    # Pivot the data to create a matrix
    pivot_data = qse_df.pivot(index='rule_number', columns='cluster_id', values='qse')
    
    fig, ax = plt.subplots()
    
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels([f'Cluster {c}' for c in pivot_data.columns])
    ax.set_yticklabels([f'Rule {r}' for r in pivot_data.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('QSE Score', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black")
    
    ax.set_title('QSE Score Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"QSE heatmap saved to {save_path}")

def plot_qse_comparison_table(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a detailed comparison table for the best rules per cluster.
    Similar to Table 2 in the paper.
    """
    # Get the best rule for each cluster (highest QSE)
    best_rules = qse_df.loc[qse_df.groupby('cluster_id')['qse'].idxmax()]
    
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in best_rules.iterrows():
        table_data.append([
            f"Cluster {row['cluster_id']}",
            f"Rule {row['rule_number']}",
            f"{row['coverage']:.3f}",
            f"{row['separation_error']:.3f}",
            f"{row['conciseness']:.3f}",
            f"{row['qse']:.3f}",
            row['num_predicates']
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Cluster', 'Rule #', 'Coverage', 'Sep. Error', 'Conciseness', 'QSE', '# Predicates'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.10, 0.12, 0.12, 0.12, 0.10, 0.14])
    
    table.auto_set_font_size(True)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternating
    colors = ['#ecf0f1', '#d5dbdb']
    for i in range(len(table_data)):
        for j in range(7):
            table[(i+1, j)].set_facecolor(colors[i % 2])
    
    plt.title('Best Rules per Cluster - QSE Comparison', pad=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"QSE comparison table saved to {save_path}")

def plot_qse_radar_chart(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create radar charts showing the three QSE components for best rules.
    """
    from math import pi
    
    # Get best rule per cluster
    best_rules = qse_df.loc[qse_df.groupby('cluster_id')['qse'].idxmax()]
    
    # Categories for radar chart
    categories = ['Coverage', '1 - Sep. Error', 'Conciseness']
    N = len(categories)
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create subplots for each cluster
    n_clusters = len(best_rules)
    fig, axes = plt.subplots(1, n_clusters, 
                             subplot_kw=dict(projection='polar'))
    
    if n_clusters == 1:
        axes = [axes]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, best_rules.iterrows())):
        # Values for this rule
        values = [
            row['coverage'],
            1 - row['separation_error'],
            row['conciseness']
        ]
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Rule {row["rule_number"]}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cluster {row["cluster_id"]}\nQSE = {row["qse"]:.3f}', 
                    pad=20, fontweight='bold')
        ax.grid(True)
        
        # Add reference circle at 0.7 (good threshold)
        ax.plot(angles, [0.7] * len(angles), 'r--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"QSE radar chart saved to {save_path}")

def plot_qse_detailed_table(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a comprehensive table with ALL rules and their metrics.
    """
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data - sorted by cluster and QSE
    qse_sorted = qse_df.sort_values(['cluster_id', 'qse'], ascending=[True, False])
    
    table_data = []
    for _, row in qse_sorted.iterrows():
        # Truncate rule text if too long
        rule_text = row['rule']
        if len(rule_text) > 60:
            rule_text = rule_text[:57] + '...'
        
        table_data.append([
            f"C{row['cluster_id']}",
            f"R{row['rule_number']}",
            rule_text,
            f"{row['coverage']:.3f}",
            f"{row['separation_error']:.3f}",
            f"{row['conciseness']:.3f}",
            f"{row['qse']:.3f}",
            str(row['num_predicates'])
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Cls', 'Rule', 'Rule Description', 'Cov', 'SepErr', 'Conc', 'QSE', '#Pred'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.05, 0.05, 0.50, 0.07, 0.08, 0.07, 0.07, 0.06])
    
    table.auto_set_font_size(True)
    table.scale(1, 2)
    
    # Color header
    for i in range(8):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by cluster
    current_cluster = None
    color_idx = 0
    cluster_colors = ['#ecf0f1', '#d5dbdb']
    
    for i, (_, row) in enumerate(qse_sorted.iterrows(), 1):
        if current_cluster != row['cluster_id']:
            current_cluster = row['cluster_id']
            color_idx = (color_idx + 1) % 2
        
        for j in range(8):
            table[(i, j)].set_facecolor(cluster_colors[color_idx])
            
            # Highlight best QSE in each cluster
            if j == 6 and i > 1:  # QSE column
                qse_val = float(table[(i, j)].get_text().get_text())
                if qse_val >= 0.7:
                    table[(i, j)].set_facecolor('#2ecc71')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif qse_val < 0.4:
                    table[(i, j)].set_facecolor('#e74c3c')
                    table[(i, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Complete QSE Analysis - All Rules', pad=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"QSE detailed table saved to {save_path}")

def plot_qse_all_rules_per_cluster(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a comprehensive plot showing all rules for each cluster with all QSE components.
    One subplot per cluster showing all its rules with Coverage, 1-SepErr, Conciseness, and QSE.
    """
    clusters = sorted(qse_df['cluster_id'].unique())
    n_clusters = len(clusters)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_clusters, 1)
    
    # Handle case of single cluster
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster_id in enumerate(clusters):
        ax = axes[idx]
        cluster_data = qse_df[qse_df['cluster_id'] == cluster_id].sort_values('rule_number')
        
        n_rules = len(cluster_data)
        x = np.arange(n_rules)
        width = 0.2
        
        # Extract metrics
        coverage = cluster_data['coverage'].values
        separation_quality = 1 - cluster_data['separation_error'].values
        conciseness = cluster_data['conciseness'].values
        qse = cluster_data['qse'].values
        rule_labels = [f"R{r}" for r in cluster_data['rule_number'].values]
        
        # Plot bars
        bars1 = ax.bar(x - 1.5*width, coverage, width, label='Coverage', 
                      color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x - 0.5*width, separation_quality, width, label='1 - Sep. Error', 
                      color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars3 = ax.bar(x + 0.5*width, conciseness, width, label='Conciseness', 
                      color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars4 = ax.bar(x + 1.5*width, qse, width, label='QSE', 
                      color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        
        # Formatting
        ax.set_xlabel('Rule Number', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'Cluster {cluster_id} - All Rules QSE Analysis ({n_rules} rules)', 
                    fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(rule_labels)
        ax.set_ylim(0, 1.2)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add horizontal reference lines
        ax.axhline(y=0.7, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Good threshold')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Fair threshold')
        
        # Add cluster statistics text
        stats_text = (f"Avg QSE: {qse.mean():.3f} | "
                     f"Best QSE: {qse.max():.3f} | "
                     f"Avg Coverage: {coverage.mean():.3f} | "
                     f"Avg Sep.Qual: {separation_quality.mean():.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QSE all rules per cluster plot saved to {save_path}")

def plot_qse_by_cluster_all_rules(qse_df: pd.DataFrame, save_path: str = None):
    """
    Create a box plot with clusters on x-axis showing the distribution of QSE scores for all rules.
    """
    clusters = sorted(qse_df['cluster_id'].unique())
    
    fig, ax = plt.subplots()
    
    # Prepare data for box plot
    data_to_plot = []
    cluster_labels = []
    n_rules_per_cluster = []
    
    for cluster_id in clusters:
        cluster_data = qse_df[qse_df['cluster_id'] == cluster_id]
        qse_values = cluster_data['qse'].values
        data_to_plot.append(qse_values)
        cluster_labels.append(f'Cluster {cluster_id}')
        n_rules_per_cluster.append(len(qse_values))
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, 
                    labels=cluster_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='darkred', 
                                  markersize=8, label='Mean'),
                    medianprops=dict(color='darkblue', linewidth=2.5),
                    boxprops=dict(facecolor='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, 
                                   markeredgecolor='darkred', alpha=0.6))
    
    # Add individual data points as scatter
    for i, (cluster_id, qse_values) in enumerate(zip(clusters, data_to_plot), 1):
        # Add jitter to x-coordinates for better visibility
        x_positions = np.random.normal(i, 0.04, size=len(qse_values))
        ax.scatter(x_positions, qse_values, alpha=0.5, s=60, color='navy', 
                  edgecolors='black', linewidth=0.5, zorder=3)
    
    # Add statistics annotations
    for i, (cluster_id, qse_values, n_rules) in enumerate(zip(clusters, data_to_plot, n_rules_per_cluster), 1):
        mean_val = np.mean(qse_values)
        median_val = np.median(qse_values)
        
        # Add text box with statistics
        stats_text = f'n={n_rules}\nμ={mean_val:.3f}\nmed={median_val:.3f}'
        ax.text(i, ax.get_ylim()[1] * 0.95, stats_text,
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', 
                        edgecolor='black', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Cluster ID', fontweight='bold')
    ax.set_ylabel('QSE Score', fontweight='bold')
    ax.set_title('Distribution of QSE Scores per Cluster (All Rules)', 
                fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    
    # Add reference lines
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, alpha=0.7, 
              label='Excellent (≥0.7)', zorder=1)
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
              label='Good (≥0.5)', zorder=1)
    ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='Fair (≥0.3)', zorder=1)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkblue', linewidth=2.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markersize=8, label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', 
               markeredgecolor='black', markersize=8, label='Individual Rules'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Excellent (≥0.7)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Good (≥0.5)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Fair (≥0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
    
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QSE by cluster (boxplot) plot saved to {save_path}")


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
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Silhouette Coefficient')
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
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text label for the value
        axes[i].text(0, value, f'{value:.2f}', ha='center', va='bottom')

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

    plt.xlabel('Cluster')
    plt.ylabel('Score')
    # plt.title('Predictive quality', fontsize=14)
    plt.xticks(x, predictive_df['Cluster'])
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
    table.auto_set_font_size(True)
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
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Coefficient of Variation (CV)')
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
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Coefficient of Variation (CV)')
    # ax.set_title('Stability of Discriminative Metrics within Rules\n(Box plots showing CV distribution per rule)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Rule quality box plot saved to {save_path}")

####################################REPRESENTATIVE METRICS QUALITY####################################

# def representative_metrics_quality_detailed(clustered_data: pd.DataFrame, clusters_insights: dict, metric_cols: list):
#     """
#     Returns both summary statistics AND individual CV values for box plots.
#     """
#     summary_results = []
#     detailed_cvs = []  # Store individual CV values
    
#     for cluster_id_str, cluster_info in clusters_insights.items():
#         cluster_id = int(cluster_id_str)
        
#         # Get selected features for this cluster
#         selected_features_raw = cluster_info['feature_selection']['selected_features']
#         selected_features = selected_features_raw
        
#         # Filter to only include metrics that exist in our metric_cols
#         selected_metrics = [f for f in selected_features if f in metric_cols]
#         non_selected_metrics = [f for f in metric_cols if f not in selected_metrics]
        
#         # Get data for this cluster only
#         cluster_mask = clustered_data['cluster'] == cluster_id
#         cluster_data = clustered_data[cluster_mask]
        
#         if len(cluster_data) < 2:
#             continue
        
#         # Calculate CV for selected metrics
#         selected_cvs = []
#         for metric in selected_metrics:
#             if metric in cluster_data.columns:
#                 mean_val = cluster_data[metric].mean()
#                 std_val = cluster_data[metric].std()
                
#                 if abs(mean_val) > 1e-9:
#                     cv = std_val / abs(mean_val)
#                     selected_cvs.append(cv)
#                     # Store individual CV value
#                     detailed_cvs.append({
#                         'Cluster': cluster_id,
#                         'Metric': metric,
#                         'Type': 'Selected',
#                         'CV': cv
#                     })
        
#         # Calculate CV for non-selected metrics
#         non_selected_cvs = []
#         for metric in non_selected_metrics:
#             if metric in cluster_data.columns:
#                 mean_val = cluster_data[metric].mean()
#                 std_val = cluster_data[metric].std()
                
#                 if abs(mean_val) > 1e-9:
#                     cv = std_val / abs(mean_val)
#                     non_selected_cvs.append(cv)
#                     # Store individual CV value
#                     detailed_cvs.append({
#                         'Cluster': cluster_id,
#                         'Metric': metric,
#                         'Type': 'Non-Selected',
#                         'CV': cv
#                     })
        
#         # Compute summary statistics
#         summary_results.append({
#             'Cluster': cluster_id,
#             'N_Workflows': len(cluster_data),
#             'N_Selected_Metrics': len(selected_metrics),
#             'N_Non_Selected_Metrics': len(non_selected_metrics),
#             'Selected_CV_Mean': np.mean(selected_cvs) if selected_cvs else np.nan,
#             'Selected_CV_Median': np.median(selected_cvs) if selected_cvs else np.nan,
#             'Selected_CV_Std': np.std(selected_cvs) if selected_cvs else np.nan,
#             'Non_Selected_CV_Mean': np.mean(non_selected_cvs) if non_selected_cvs else np.nan,
#             'Non_Selected_CV_Median': np.median(non_selected_cvs) if non_selected_cvs else np.nan,
#             'Non_Selected_CV_Std': np.std(non_selected_cvs) if non_selected_cvs else np.nan,
#             'CV_Difference': (np.mean(non_selected_cvs) - np.mean(selected_cvs)) if (selected_cvs and non_selected_cvs) else np.nan
#         })
    
#     summary_df = pd.DataFrame(summary_results)
#     detailed_df = pd.DataFrame(detailed_cvs)
    
#     return summary_df, detailed_df

def representative_metrics_quality_detailed(clustered_data: pd.DataFrame, clusters_insights: dict, metric_cols: list):
    """
    Returns both summary statistics AND individual CV values for box plots.
    """
    summary_results = []
    detailed_cvs = []  # Store individual CV values
    
    for cluster_id_str, cluster_info in clusters_insights.items():
        cluster_id = int(cluster_id_str)
        
        # Get selected features for this cluster
        selected_features_raw = cluster_info['feature_selection']['selected_features']
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

def plot_representative_metrics_cv_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Create a simple bar plot comparing CV for selected vs non-selected metrics per cluster.
    """
    fig, ax = plt.subplots()
    
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
                       ha='center', va='bottom')
    
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
    fig, ax = plt.subplots(figsize=(8, 6.2))

    bp = ax.boxplot(all_data, 
                    positions=positions,
                    patch_artist=True,
                    showmeans=False,
                    widths=0.4)

    # Colorblind-friendly colors: blue for good (representative), orange for bad (other)
    # These colors are distinguishable for all types of colorblindness and in grayscale
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
    ax.set_ylabel('CV Values', fontweight='bold', fontsize=FONT_SIZE['ylabel'])
    ax.tick_params(axis='y', labelsize=FONT_SIZE['ytick'])
    # ax.set_title('Selected vs Non-Selected CV Distribution by Cluster - Adult Dataset (Real Data)', 
    #              fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add legend with colorblind-friendly colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_good, edgecolor='black', linewidth=1.2, label='Representative Metrics'),
                    Patch(facecolor=color_bad, edgecolor='black', linewidth=1.2, label='Other Metrics')]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE['legend'], frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path)
        print(f"Representative metrics CV boxplot saved to {save_path}")
    
    return fig, ax

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
    
    # Generate detailed boxplot for representative metrics quality
    summary_df, detailed_df = representative_metrics_quality_detailed(clustered_data, clusters_insights, metric_cols)
    detailed_df.to_csv(os.path.join(result_dir_representative_quality, "representative_quality_detailed.csv"), index=False)
    plot_representative_metrics_cv_boxplot(detailed_df, save_path=os.path.join(result_dir_representative_quality, "cv_boxplot.png"))
    print(f"Box plots created with data from {len(detailed_df)} individual CV values!")
    
    ## Explanation Quality Score (QSE)
    os.makedirs(result_dir_qse, exist_ok=True)
    
    if rules_list:
        print("\n" + "="*80)
        print("CALCULATING EXPLANATION QUALITY SCORES (QSE)")
        print("="*80)
        
        # Load workflow data for QSE calculation
        raw_data_full = pd.read_csv(f"{path}/workflows.csv")
        cluster_labels = pd.read_csv(f"{path}/workflows_clustered.csv")[['workflowId', 'cluster']]
        all_data_with_clusters = pd.merge(raw_data_full, cluster_labels, on='workflowId')
        all_data_with_clusters.columns = [c.replace(' ', '_') for c in all_data_with_clusters.columns]
        
        # Calculate QSE for all rules
        qse_df = calculate_all_qse_metrics(rules_df, all_data_with_clusters)
        
        # Save QSE metrics
        qse_df.to_csv(os.path.join(result_dir_qse, "qse_metrics.csv"), index=False)
        print(f"\n✓ QSE metrics saved to {result_dir_qse}/qse_metrics.csv")
        
        # Display summary statistics
        print("\nQSE Summary Statistics:")
        print(qse_df.groupby('cluster_id')[['coverage', 'separation_error', 'conciseness', 'qse']].mean())
        
        print("\nBest Rule per Cluster (by QSE):")
        best_rules = qse_df.loc[qse_df.groupby('cluster_id')['qse'].idxmax()]
        print(best_rules[['cluster_id', 'rule_number', 'coverage', 'separation_error', 'conciseness', 'qse']])
        
        # Generate plots
        plot_qse_metrics(qse_df, save_path=os.path.join(result_dir_qse, "qse_components.png"))
        plot_qse_overall(qse_df, save_path=os.path.join(result_dir_qse, "qse_overall.png"))
        plot_qse_heatmap(qse_df, save_path=os.path.join(result_dir_qse, "qse_heatmap.png"))
        plot_qse_comparison_table(qse_df, save_path=os.path.join(result_dir_qse, "qse_comparison_table.png"))
        plot_qse_radar_chart(qse_df, save_path=os.path.join(result_dir_qse, "qse_radar_chart.png"))
        plot_qse_detailed_table(qse_df, save_path=os.path.join(result_dir_qse, "qse_detailed_table.png"))
        plot_qse_all_rules_per_cluster(qse_df, save_path=os.path.join(result_dir_qse, "qse_all_rules_per_cluster.png"))
        plot_qse_by_cluster_all_rules(qse_df, save_path=os.path.join(result_dir_qse, "qse_by_cluster_all_rules.png"))
        
        print(f"\n✓ All QSE plots saved to {result_dir_qse}/")
        print(f"  - qse_components.png: Coverage, Separation, Conciseness breakdown")
        print(f"  - qse_overall.png: Overall QSE scores by cluster and rule")
        print(f"  - qse_heatmap.png: QSE score matrix visualization")
        print(f"  - qse_comparison_table.png: Best rules comparison table")
        print(f"  - qse_radar_chart.png: Radar charts for best rules")
        print(f"  - qse_detailed_table.png: Complete metrics for all rules")
        print(f"  - qse_all_rules_per_cluster.png: All rules combined per cluster with statistics")
        print(f"  - qse_by_cluster_all_rules.png: Clusters on x-axis with all rules as bars")
        
    else:
        print("Warning: No rules available. Skipping QSE calculation.")



