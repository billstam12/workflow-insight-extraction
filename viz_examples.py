"""
Visualization Examples for Paper Figures
=========================================
Extracted visualization functions to create customizable examples for paper figures.
Choose your own metrics and data to showcase in papers.

This module provides:
1. HIGH Metrics Radar Chart
2. LOW Metrics Radar Chart
3. Feature Correlation Networks
4. Parallel Coordinates Plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# RADAR CHART 1: HIGH METRICS (Higher is Better)
# ============================================================================

def plot_high_metrics_radar(
    metrics_dict,
    comparison_data_list=None,
    title="HIGH Metrics Profile",
    color_main='#2ca02c',
    figsize=(10, 8),
    output_file=None
):
    """
    Create a radar chart for HIGH metrics (higher is better).
    
    Args:
        metrics_dict: Dict like {'Accuracy': 0.85, 'Precision': 0.92, 'F1': 0.88}
        comparison_data_list: Optional list of dicts with same keys for comparison
        title: Chart title
        color_main: Color for main profile
        figsize: Figure size
        output_file: Path to save figure
    
    Example:
        high_metrics = {'Accuracy': 0.85, 'Precision': 0.92, 'F1-Score': 0.88, 'AUC': 0.91}
        plot_high_metrics_radar(
            high_metrics,
            title="Cluster HIGH Metrics",
            color_main='#2ca02c',
            output_file='high_radar.png'
        )
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    if not metrics_dict:
        print("Error: No metrics provided!")
        return
    
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    N = len(metric_names)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_plot = angles + angles[:1]
    
    # Plot comparison data if provided
    if comparison_data_list:
        for comp_data in comparison_data_list:
            comp_values = [comp_data.get(m, 0.5) for m in metric_names]
            comp_values_plot = comp_values + comp_values[:1]
            ax.plot(angles_plot, comp_values_plot, 'o-', linewidth=1.5,
                   color='gray', alpha=0.3, markersize=4)
    
    # Plot main profile
    values_plot = metric_values + metric_values[:1]
    ax.plot(angles_plot, values_plot, 'o-', linewidth=3.5, color=color_main,
           markersize=10, zorder=10)
    ax.fill(angles_plot, values_plot, alpha=0.25, color=color_main)
    
    # Labels with up arrow indicator
    labels = [f"↑ {m}" for m in metric_names]
    
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=11, wrap=True)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.5', '0.75'], size=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title, size=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved HIGH metrics radar to {output_file}")
    
    return fig, ax


# ============================================================================
# RADAR CHART 2: LOW METRICS (Lower is Better)
# ============================================================================

def plot_low_metrics_radar(
    metrics_dict,
    comparison_data_list=None,
    title="LOW Metrics Profile",
    color_main='#d62728',
    figsize=(10, 8),
    output_file=None
):
    """
    Create a radar chart for LOW metrics (lower is better).
    
    Args:
        metrics_dict: Dict like {'Latency': 0.2, 'Error': 0.1, 'Memory': 0.15}
        comparison_data_list: Optional list of dicts with same keys for comparison
        title: Chart title
        color_main: Color for main profile
        figsize: Figure size
        output_file: Path to save figure
    
    Example:
        low_metrics = {'Error Rate': 0.08, 'Latency': 0.15, 'Memory': 0.12}
        plot_low_metrics_radar(
            low_metrics,
            title="Cluster LOW Metrics",
            color_main='#d62728',
            output_file='low_radar.png'
        )
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    if not metrics_dict:
        print("Error: No metrics provided!")
        return
    
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    N = len(metric_names)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_plot = angles + angles[:1]
    
    # Plot comparison data if provided
    if comparison_data_list:
        for comp_data in comparison_data_list:
            comp_values = [comp_data.get(m, 0.5) for m in metric_names]
            comp_values_plot = comp_values + comp_values[:1]
            ax.plot(angles_plot, comp_values_plot, 'o-', linewidth=1.5,
                   color='gray', alpha=0.3, markersize=4)
    
    # Plot main profile
    values_plot = metric_values + metric_values[:1]
    ax.plot(angles_plot, values_plot, 'o-', linewidth=3.5, color=color_main,
           markersize=10, zorder=10)
    ax.fill(angles_plot, values_plot, alpha=0.25, color=color_main)
    
    # Labels with down arrow indicator
    labels = [f"↓ {m}" for m in metric_names]
    
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=11, wrap=True)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.5', '0.75'], size=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title, size=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved LOW metrics radar to {output_file}")
    
    return fig, ax


# ============================================================================
# FEATURE CORRELATION NETWORK
# ============================================================================

def plot_feature_network(
    features_dict,
    correlations_list=None,
    alternatives_list=None,
    tradeoffs_list=None,
    node_color='#1f77b4',
    figsize=(12, 10),
    output_file=None,
    title="Feature Relationships"
):
    """
    Create a network graph showing feature relationships and correlations.
    
    Args:
        features_dict: Dict like {'Feature1': 0.75, 'Feature2': 0.92}
                      where values are node sizes/importance
        correlations_list: List of tuples [('Feature1', 'Feature2', 0.85), ...]
        alternatives_list: List of tuples for alternative relationships
        tradeoffs_list: List of tuples for trade-off relationships
        node_color: Color for nodes
        figsize: Figure size
        output_file: Path to save figure
        title: Chart title
    
    Example:
        features = {
            'Accuracy': 0.9,
            'Precision': 0.85,
            'F1-Score': 0.88,
            'Latency': 0.2
        }
        correlations = [
            ('Accuracy', 'Precision', 0.92),
            ('Precision', 'F1-Score', 0.88),
        ]
        alternatives = [
            ('Accuracy', 'Recall', 0.85),
        ]
        tradeoffs = [
            ('Accuracy', 'Latency', -0.75),
        ]
        
        plot_feature_network(
            features,
            correlations,
            alternatives,
            tradeoffs,
            output_file='network_example.png'
        )
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes from features
    for feature, importance in features_dict.items():
        G.add_node(feature, importance=importance)
    
    # Add edges
    edge_colors = []
    edge_widths = []
    edge_styles = []
    
    if correlations_list:
        for feat1, feat2, strength in correlations_list:
            if feat1 in G.nodes() and feat2 in G.nodes():
                G.add_edge(feat1, feat2, weight=abs(strength), type='correlation')
                edge_colors.append('#1f77b4')  # Blue for correlations
                edge_widths.append(2 + abs(strength) * 3)
                edge_styles.append('-')
    
    if alternatives_list:
        for feat1, feat2, strength in alternatives_list:
            if feat1 in G.nodes() and feat2 in G.nodes():
                G.add_edge(feat1, feat2, weight=abs(strength), type='alternative')
                edge_colors.append('#ff7f0e')  # Orange for alternatives
                edge_widths.append(1.5 + abs(strength) * 2)
                edge_styles.append('--')
    
    if tradeoffs_list:
        for feat1, feat2, strength in tradeoffs_list:
            if feat1 in G.nodes() and feat2 in G.nodes():
                G.add_edge(feat1, feat2, weight=abs(strength), type='tradeoff')
                edge_colors.append('#d62728')  # Red for trade-offs
                edge_widths.append(1.5 + abs(strength) * 2)
                edge_styles.append(':')
    
    # Compute node sizes
    node_sizes = [features_dict.get(node, 0.5) * 3000 for node in G.nodes()]
    
    # Layout using spring algorithm
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color,
                          alpha=0.9, ax=ax, edgecolors='black', linewidths=2)
    
    # Draw edges grouped by style
    for style, width, color in [('-', 2, '#1f77b4'), ('--', 1.5, '#ff7f0e'), (':', 1.5, '#d62728')]:
        edges_to_draw = [(u, v, d) for u, v, d in G.edges(data=True)
                        if style == '-' and d.get('type') == 'correlation' or
                           style == '--' and d.get('type') == 'alternative' or
                           style == ':' and d.get('type') == 'tradeoff']
        
        if edges_to_draw:
            nx.draw_networkx_edges(G, pos, [(u, v) for u, v, _ in edges_to_draw],
                                  width=width, style=style, edge_color=color,
                                  alpha=0.6, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    
    # Create legend
    corr_line = mlines.Line2D([], [], color='#1f77b4', linewidth=2,
                             linestyle='-', label='Correlated')
    alt_line = mlines.Line2D([], [], color='#ff7f0e', linewidth=2,
                            linestyle='--', label='Alternative')
    trade_line = mlines.Line2D([], [], color='#d62728', linewidth=2,
                              linestyle=':', label='Trade-off')
    
    ax.legend(handles=[corr_line, alt_line, trade_line],
             loc='upper left', fontsize=11, framealpha=0.95)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved network graph to {output_file}")
    
    return fig, ax


# ============================================================================
# PARALLEL COORDINATES PLOT
# ============================================================================

def plot_parallel_coordinates(
    data_df,
    column_order=None,
    color_column=None,
    highlight_rows=None,
    color_map='viridis',
    figsize=(14, 6),
    output_file=None,
    title="Parallel Coordinates"
):
    """
    Create a parallel coordinates plot for multivariate data.
    
    Args:
        data_df: DataFrame with normalized values (0-1)
        column_order: List of column names in desired order
        color_column: Column name to use for coloring lines
        highlight_rows: List of row indices to highlight with darker color
        color_map: Colormap name
        figsize: Figure size
        output_file: Path to save figure
        title: Chart title
    
    Example:
        data = pd.DataFrame({
            'LearningRate': np.random.uniform(0, 1, 50),
            'BatchSize': np.random.uniform(0, 1, 50),
            'Dropout': np.random.uniform(0, 1, 50),
            'Accuracy': np.random.uniform(0.5, 1, 50),
            'F1': np.random.uniform(0.5, 1, 50)
        })
        
        plot_parallel_coordinates(
            data,
            column_order=['LearningRate', 'BatchSize', 'Dropout', 'Accuracy', 'F1'],
            color_column='Accuracy',
            highlight_rows=[0, 1],
            output_file='parallel_coords_example.png'
        )
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    if column_order is None:
        column_order = list(data_df.columns)
    
    # Normalize each column to [0, 1] if not already
    data_norm = data_df[column_order].copy()
    for col in column_order:
        col_min = data_norm[col].min()
        col_max = data_norm[col].max()
        if col_max > col_min:
            data_norm[col] = (data_norm[col] - col_min) / (col_max - col_min)
    
    # Get color values if specified
    if color_column and color_column in data_norm.columns:
        color_values = data_norm[color_column].values
        cmap = plt.cm.get_cmap(color_map)
        colors_arr = cmap(color_values)
    else:
        colors_arr = ['#1f77b4'] * len(data_norm)
    
    # Plot each row as a line
    x_pos = np.arange(len(column_order))
    
    for idx, (_, row) in enumerate(data_norm.iterrows()):
        if highlight_rows and idx in highlight_rows:
            ax.plot(x_pos, row.values, color=colors_arr[idx], alpha=0.8,
                   linewidth=2.5, zorder=100)
        else:
            ax.plot(x_pos, row.values, color=colors_arr[idx], alpha=0.1,
                   linewidth=1)
    
    # If highlighting rows, also plot mean line
    if highlight_rows:
        mean_line = data_norm.mean()
        ax.plot(x_pos, mean_line.values, color='#d62728', alpha=0.9,
               linewidth=3, marker='o', markersize=8, label='Mean', zorder=200)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(column_order, fontsize=10, rotation=45, ha='right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    if highlight_rows:
        ax.legend(fontsize=10, loc='upper right')
    
    # Add colorbar if color_column specified
    if color_column and color_column in data_norm.columns:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(color_map),
                                   norm=plt.Normalize(vmin=color_values.min(),
                                                     vmax=color_values.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(color_column, fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved parallel coordinates to {output_file}")
    
    return fig, ax


# ============================================================================
# COMPLETE EXAMPLE: GENERATE ALL VISUALIZATION EXAMPLES
# ============================================================================

def generate_all_examples(output_dir='results/paper_figures'):
    """
    Generate all example visualizations for your paper.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION EXAMPLES FOR PAPER")
    print("="*80 + "\n")
    
    # ========== EXAMPLE 1A: HIGH METRICS RADAR CHART ==========
    print("1a. Creating HIGH Metrics Radar Chart Example...")
    high_metrics = {
        'Accuracy': 0.89,
        'Recall': 0.92,
        'F1-Score': 0.87,
        'AUC': 0.91
    }
    
    comparison_high = {
        'Accuracy': 0.75,
        'Recall': 0.78,
        'F1-Score': 0.76,
        'AUC': 0.77
    }
    
    # high_metrics = {
    #     'Precision': 0.92,
    #     'Overall Equality': 0.89,
    #     'Statistical Parity': 0.87,
    #     'Treatment Equality': 0.91
    # }
    
    # comparison_high = {
    #     'Precision': 0.78,
    #     'Overall Equality': 0.75,
    #     'Statistical Parity': 0.76,
    #     'Treatment Equality': 0.77
    # }
    

    plot_high_metrics_radar(
        high_metrics,
        comparison_data_list=[comparison_high],
        title="HIGH Metrics Profile - Example Cluster",
        color_main='#2ca02c',
        output_file=f'{output_dir}/01_high_metrics_radar.png'
    )
    plt.close()
    
    # ========== EXAMPLE 1B: LOW METRICS RADAR CHART ==========
    print("1b. Creating LOW Metrics Radar Chart Example...")
    low_metrics = {
        'Precision': 0.08,
        'Fairness': 0.15,
        'Equality': 0.12
    }
    
    comparison_low = {
        'Precision': 0.20,
        'Fairness': 0.35,
        'Equality': 0.28
    }
    
    plot_low_metrics_radar(
        low_metrics,
        comparison_data_list=[comparison_low],
        title="LOW Metrics Profile - Example Cluster",
        color_main='#d62728',
        output_file=f'{output_dir}/01b_low_metrics_radar.png'
    )
    plt.close()
    
    # ========== EXAMPLE 2: FEATURE NETWORK ==========
    print("2. Creating Feature Network Graph Example...")
    features = {
        'Accuracy': 0.9,
        'Precision': 0.85,
        'F1-Score': 0.88,
        'Recall': 0.82,
        'AUC': 0.91,
        'Latency': 0.3
    }
    
    correlations = [
        ('Accuracy', 'Precision', 0.92),
        ('Precision', 'F1-Score', 0.88),
        ('F1-Score', 'Recall', 0.85),
        ('Accuracy', 'AUC', 0.89),
    ]
    
    alternatives = [
        ('Accuracy', 'Balanced Accuracy', 0.85),
        ('Precision', 'Sensitivity', 0.80),
    ]
    
    tradeoffs = [
        ('Accuracy', 'Latency', -0.75),
        ('Precision', 'Latency', -0.70),
    ]
    
    plot_feature_network(
        features,
        correlations,
        None,
        tradeoffs,
        node_color='#1f77b4',
        output_file=f'{output_dir}/02_feature_network.png',
        title="Feature Relationships & Trade-offs"
    )
    plt.close()
    
    # ========== EXAMPLE 3: PARALLEL COORDINATES ==========
    print("3. Creating Parallel Coordinates Example...")
    np.random.seed(42)
    data = pd.DataFrame({
        'Learning Rate': np.random.uniform(0, 1, 50),
        'Batch Size': np.random.uniform(0, 1, 50),
        'Dropout': np.random.uniform(0, 1, 50),
        'L2 Reg': np.random.uniform(0, 1, 50),
        'Accuracy': np.random.uniform(0.5, 1, 50),
        'F1 Score': np.random.uniform(0.4, 0.95, 50)
    })
    
    # Highlight rows with high accuracy
    best_rows = data.nlargest(3, 'Accuracy').index.tolist()
    
    plot_parallel_coordinates(
        data,
        column_order=['Learning Rate', 'Batch Size', 'Dropout', 'L2 Reg', 'Accuracy', 'F1 Score'],
        color_column='Accuracy',
        highlight_rows=best_rows,
        color_map='YlOrRd',
        output_file=f'{output_dir}/03_parallel_coordinates.png',
        title="Configuration Space Analysis (Best Performers Highlighted)"
    )
    plt.close()
    
    print("\n" + "="*80)
    print("✓ All examples generated successfully!")
    print(f"✓ Output directory: {output_dir}/")
    print("="*80 + "\n")
    print("Generated files:")
    print("  - 01_high_metrics_radar.png: Customize HIGH metrics (↑ higher is better)")
    print("  - 01b_low_metrics_radar.png: Customize LOW metrics (↓ lower is better)")
    print("  - 02_feature_network.png: Add/remove features and relationships")
    print("  - 03_parallel_coordinates.png: Add configurations and metrics")
    print("\nEdit the generate_all_examples() function to customize for your paper!\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate all example visualizations
    generate_all_examples('results/paper_figures')
    
    # You can also create individual visualizations:
    # 
    # example_high = {'Metric1': 0.85, 'Metric2': 0.92}
    # plot_high_metrics_radar(example_high, title="My HIGH Metrics")
    #
    # example_low = {'Metric3': 0.2, 'Metric4': 0.15}
    # plot_low_metrics_radar(example_low, title="My LOW Metrics")
    # 
    # plt.show()
