"""
Workflow Cluster Dashboards
============================
Per-cluster comprehensive visualization dashboards combining:
- Representative metric profiles (radar chart)
- Trade-off correlations (heatmap)
- Configuration patterns (parallel coordinates)
- Cluster statistics and decision rules
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.patches import Patch
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_cluster_data(data_folder='data/workflows'):
    """Load all clustering and insights data."""
    prefix = os.path.basename(data_folder)
    
    df_clustered = pd.read_csv(os.path.join(data_folder, "workflows_clustered.csv"))
    
    with open(os.path.join(data_folder, "clusters_comprehensive_insights.json"), 'r') as f:
        insights = json.load(f)
    
    try:
        df_model_eval = pd.read_csv(os.path.join(data_folder, "workflows_model_evaluation_summary.csv"))
    except FileNotFoundError:
        df_model_eval = None
    
    return df_clustered, insights, df_model_eval


def extract_shap_features_and_tradeoffs(df_clustered, cluster_id, insights, use_high_shap_only=True):
    """
    Extract representative features for a cluster.
    
    Args:
        use_high_shap_only: If True, use only high SHAP features. 
                           If False, use all selected features.
    
    Returns dict with:
    - shap_features: List of representative features
    - values: Dict mapping feature name to normalized value
    - categories: Dict mapping feature name to category (HIGH/LOW/MID)
    - high_features: List of features categorized as HIGH
    - low_features: List of features categorized as LOW
    """
    cluster_insights = insights.get(str(cluster_id), {})
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    
    shap_features = []  # Initialize for use in all branches
    
    if use_high_shap_only:
        # Extract high SHAP features - now it's a dict with 'features' key
        if 'high_shap_features' in cluster_insights:
            shap_info = cluster_insights['high_shap_features']
            if isinstance(shap_info, dict):
                shap_features = shap_info.get('features', [])
                if isinstance(shap_features, str):
                    shap_features = [f.strip() for f in shap_features.split(',') if f.strip()]
            elif isinstance(shap_info, str):
                shap_features = [f.strip() for f in shap_info.split(',') if f.strip()]
            elif isinstance(shap_info, list):
                shap_features = [str(f).strip() for f in shap_info if f]
        
        all_features = list(set(shap_features))
        feature_stats_source = cluster_insights.get('high_shap_features', {}).get('feature_statistics', {})
    else:
        # Use all selected features from feature_selection
        all_features = []
        if 'feature_selection' in cluster_insights:
            feat_info = cluster_insights['feature_selection']
            if isinstance(feat_info, dict):
                selected_feats = feat_info.get('selected_features', [])
                if isinstance(selected_feats, str):
                    all_features = [f.strip() for f in selected_feats.split(',') if f.strip()]
                elif isinstance(selected_feats, list):
                    all_features = [str(f).strip() for f in selected_feats if f]
        
        feature_stats_source = cluster_insights.get('feature_selection', {}).get('feature_statistics', {})
    
    if not all_features:
        # Fallback if no features found
        return {
            'shap_features': [],
            'all_features': [],
            'values': {},
            'categories': {},
            'high_features': [],
            'low_features': [],
            'cluster_size': len(cluster_data),
            'cluster_proportion': len(cluster_data) / len(df_clustered)
        }
    
    # Get values and categories from JSON feature_statistics
    values = {}
    categories = {}
    high_features = []
    low_features = []
    
    for feature in all_features:
        if feature in df_clustered.columns:
            col_min = df_clustered[feature].min()
            col_max = df_clustered[feature].max()
            cluster_mean = cluster_data[feature].mean()
            
            # Normalize to [0, 1]
            normalized_value = (cluster_mean - col_min) / (col_max - col_min + 1e-8)
            values[feature] = normalized_value
            
            # Get value_category from JSON (already computed)
            feature_stat = feature_stats_source.get(feature, {})
            value_cat = feature_stat.get('value_category', 'mid').upper()
            categories[feature] = value_cat
            
            # Categorize as HIGH or LOW
            if value_cat == 'HIGH':
                high_features.append(feature)
            elif value_cat == 'LOW':
                low_features.append(feature)
    
    return {
        'shap_features': [f for f in shap_features if f in all_features],
        'all_features': all_features,
        'values': values,
        'categories': categories,
        'high_features': high_features,
        'low_features': low_features,
        'cluster_size': len(cluster_data),
        'cluster_proportion': len(cluster_data) / len(df_clustered)
    }


def get_normalized_feature_value(df_clustered, cluster_id, feature_name):
    """
    Get the normalized value (0-1) of a feature for a cluster.
    Uses min-max normalization across the entire dataset.
    """
    if feature_name not in df_clustered.columns:
        return 0.5
    
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    
    col_min = df_clustered[feature_name].min()
    col_max = df_clustered[feature_name].max()
    cluster_mean = cluster_data[feature_name].mean()
    
    # Normalize to [0, 1]
    normalized_value = (cluster_mean - col_min) / (col_max - col_min + 1e-8)
    return normalized_value


def extract_alternatives_and_tradeoffs(df_clustered, cluster_insights, shap_data, category_type='HIGH', all_selected_features=None):
    """
    Extract alternative features (removed due to correlation) and trade-offs for a given category.
    
    Args:
        df_clustered: Clustered dataframe
        cluster_insights: Cluster insights dictionary
        shap_data: SHAP data dictionary with features and categories
        category_type: 'HIGH' or 'LOW' to extract alternatives for this category
        all_selected_features: All selected features list
    
    Returns:
        List of dictionaries with alternative/trade-off features and their metadata
    """
    categories = shap_data['categories']
    
    # Get features that match our category
    if category_type == 'HIGH':
        primary_features = [f for f in shap_data['all_features'] if categories.get(f) == 'HIGH']
        opposite_category = 'LOW'
    else:
        primary_features = [f for f in shap_data['all_features'] if categories.get(f) == 'LOW']
        opposite_category = 'HIGH'
    
    if not primary_features:
        return []
    
    results = []
    
    # Get removed features (correlations to selected features)
    correlation_analysis = cluster_insights.get('correlation_analysis', {})
    removed_features = correlation_analysis.get('removed_features', {})
    
    if removed_features:
        for feat, details in removed_features.items():
            related_to = details.get('related_to', '')
            strength = details.get('max_relationship', 0)
            
            # Only include if related_to is one of our primary category features
            if related_to in primary_features:
                results.append({
                    'feature': feat,
                    'strength': strength,
                    'related_to': related_to,
                    'type': 'Alternative'
                })
    
    # Get trade-off analysis: we want trade-offs FROM opposite category features
    trade_off_analysis = cluster_insights.get('trade_off_analysis', {})
    strong_tradeoffs = trade_off_analysis.get('strong_tradeoffs', [])
    
    if strong_tradeoffs and primary_features:
        for trade in strong_tradeoffs:
            m1 = trade.get('metric_1', '')
            m2 = trade.get('metric_2', '')
            corr = abs(trade.get('actual_correlation', 0))
            
            # We want trade-offs from opposite category
            # If m1 is our primary feature and m2 is not selected, it's a trade-off from opposite
            if m1 in primary_features and m2 not in shap_data['all_features']:
                results.append({
                    'feature': m2,
                    'strength': corr,
                    'related_to': m1,
                    'type': 'Trade-off'
                })
            # If m2 is our primary feature and m1 is not selected, it's a trade-off from opposite
            elif m2 in primary_features and m1 not in shap_data['all_features']:
                results.append({
                    'feature': m1,
                    'strength': corr,
                    'related_to': m2,
                    'type': 'Trade-off'
                })
    
    return results


def extract_cluster_metrics_summary(df_clustered, cluster_id, insights, n_top_metrics=6):
    """
    Extract key metrics for a cluster from insights JSON.
    Uses feature_statistics with high/low/mid categorization from previous analysis.
    Returns metrics with their value categories and distinctiveness scores.
    """
    cluster_insights = insights.get(str(cluster_id), {})
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    
    # Get feature statistics from insights
    feature_stats = cluster_insights.get('feature_selection', {}).get('feature_statistics', {})
    
    if not feature_stats:
        # Fallback to old method if stats not available
        numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
        config_cols = ['workflowId', 'cluster', 'criterion', 'fairness method', 
                       'max depth', 'n estimators', 'normalization', 'random state',
                       'learning_curve_train_score', 'system_cpu_utilization_percentage']
        metric_cols = [col for col in numeric_cols if col not in config_cols]
        
        cluster_means = cluster_data[metric_cols].mean()
        global_means = df_clustered[metric_cols].mean()
        global_std = df_clustered[metric_cols].std()
        
        cluster_normalized = (cluster_means - global_means) / (global_std + 1e-8)
        cluster_normalized = (cluster_normalized - cluster_normalized.min()) / (cluster_normalized.max() - cluster_normalized.min() + 1e-8)
        
        top_metrics = cluster_normalized.abs().nlargest(n_top_metrics).index.tolist()
        
        return {
            'metrics': top_metrics,
            'values': cluster_normalized[top_metrics].values,
            'mean_values': cluster_means[top_metrics].values,
            'cluster_size': len(cluster_data),
            'cluster_proportion': len(cluster_data) / len(df_clustered),
            'categories': {m: 'MID' for m in top_metrics},
            'distinctiveness': {m: 0.5 for m in top_metrics}
        }
    
    # Extract metrics with their statistics, sorted by distinctiveness
    metrics_data = []
    for metric, stats in feature_stats.items():
        if 'distinctiveness_score' in stats:
            metrics_data.append({
                'metric': metric,
                'distinctiveness': stats['distinctiveness_score'],
                'category': stats.get('value_category', 'mid').upper(),
                'cluster_mean': stats.get('cluster_mean', 0),
                'cluster_std': stats.get('cluster_std', 0),
                'normalized_mean': stats.get('cluster_mean', 0)  # Already standardized in JSON
            })
    
    # Sort by distinctiveness and get top N
    metrics_data.sort(key=lambda x: x['distinctiveness'], reverse=True)
    metrics_data = metrics_data[:n_top_metrics]
    
    top_metrics = [m['metric'] for m in metrics_data]
    values = np.array([m['normalized_mean'] for m in metrics_data])
    # Normalize to [0, 1] for radar visualization
    values_normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
    
    return {
        'metrics': top_metrics,
        'values': values_normalized,
        'mean_values': np.array([m['cluster_mean'] for m in metrics_data]),
        'cluster_size': len(cluster_data),
        'cluster_proportion': len(cluster_data) / len(df_clustered),
        'categories': {m['metric']: m['category'] for m in metrics_data},
        'distinctiveness': {m['metric']: m['distinctiveness'] for m in metrics_data}
    }


# ============================================================================
# NETWORK GRAPH VISUALIZATION (Correlations & Feature Relationships)
# ============================================================================

def plot_feature_correlation_network(ax, df_clustered, cluster_id, cluster_insights, shap_data, 
                                     high_alternatives, low_alternatives, cluster_color):
    """
    Create a network graph showing feature relationships.
    
    Node types:
    - Selected features (diamond, blue) - both HIGH and LOW combined
    - Alternative features (circle, gray)
    - Trade-off features (square, orange)
    
    Edge types:
    - Blue solid: correlation between features
    - Gray dashed: alternative relationships (feature removed due to correlation)
    - Orange dotted: trade-off relationships
    
    Only shows features that have relationships (no isolated nodes).
    """
    G = nx.Graph()
    
    # Collect all selected features (both HIGH and LOW) - treat them the same
    selected_features = set(shap_data['high_features']) | set(shap_data['low_features'])
    
    # Add selected features as nodes
    for feat in selected_features:
        G.add_node(feat)
    
    # Add alternative features (keep only top by strength to avoid crowding)
    alternatives_features = set()
    alt_list = [alt for alt in high_alternatives + low_alternatives if alt['type'] == 'Alternative']
    # Sort by strength and keep top 10 alternatives
    alt_list.sort(key=lambda x: x['strength'], reverse=True)
    for alt in alt_list[:10]:
        feat = alt['feature']
        related = alt['related_to']
        
        if related in selected_features:
            alternatives_features.add(feat)
            G.add_node(feat)
            
            # Add edge
            G.add_edge(feat, related, weight=alt['strength'], edge_type='alternative')
    
    # Add trade-off features (keep only top by correlation to avoid crowding)
    tradeoff_features = set()
    trade_off_analysis = cluster_insights.get('trade_off_analysis', {})
    strong_tradeoffs = trade_off_analysis.get('strong_tradeoffs', [])
    
    # Sort by correlation strength and keep top 10 tradeoffs
    strong_tradeoffs_sorted = sorted(strong_tradeoffs, 
                                     key=lambda x: abs(x.get('actual_correlation', 0)), 
                                     reverse=True)
    
    for trade in strong_tradeoffs_sorted[:10]:
        m1 = trade.get('metric_1', '')
        m2 = trade.get('metric_2', '')
        corr = abs(trade.get('actual_correlation', 0))
        
        # Check if either is a selected feature
        if m1 in selected_features and m2 not in selected_features:
            if m2:
                tradeoff_features.add(m2)
                G.add_node(m2)
                G.add_edge(m1, m2, weight=corr, edge_type='tradeoff')
        
        elif m2 in selected_features and m1 not in selected_features:
            if m1:
                tradeoff_features.add(m1)
                G.add_node(m1)
                G.add_edge(m1, m2, weight=corr, edge_type='tradeoff')
    
    # Add correlation edges between selected features
    correlation_analysis = cluster_insights.get('correlation_analysis', {})
    correlations = correlation_analysis.get('correlations', {})
    
    if correlations:
        for pair_key, corr_val in correlations.items():
            try:
                f1, f2 = pair_key.split(' <-> ')
                if f1 in G.nodes() and f2 in G.nodes():
                    # Only add if both are selected features (internal correlations)
                    if (f1 in selected_features and f2 in selected_features and 
                        not (f1, f2) in [(e[0], e[1]) for e in G.edges()]):
                        G.add_edge(f1, f2, weight=abs(corr_val), edge_type='correlation')
            except:
                pass
    
    # Remove isolated nodes (features with no connections)
    nodes_to_remove = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No feature relationships', ha='center', va='center',
               transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return
    
    # Layout - use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges by type
    edge_types = {
        'correlation': [],
        'alternative': [],
        'tradeoff': []
    }
    
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'correlation')
        edge_types[edge_type].append((u, v))
    
    # Draw correlation edges (blue solid)
    for u, v in edge_types['correlation']:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, 'b-', alpha=0.4, linewidth=2, zorder=1)
    
    # Draw alternative edges (gray dashed)
    for u, v in edge_types['alternative']:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Draw trade-off edges (orange dotted)
    for u, v in edge_types['tradeoff']:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color='#ff7f0e', linestyle=':', alpha=0.6, linewidth=2, zorder=1)
    
    # Draw selected feature nodes (blue diamonds)
    for node in selected_features:
        if node in pos:
            ax.scatter(pos[node][0], pos[node][1], s=800, 
                      c='#1f77b4', marker='D', edgecolors='black', linewidth=1.5, 
                      zorder=5, alpha=0.85)
    
    # Draw alternative nodes (gray circles)
    for node in alternatives_features:
        if node in pos:
            ax.scatter(pos[node][0], pos[node][1], s=600, 
                      c='#7f7f7f', marker='o', edgecolors='black', linewidth=1, 
                      zorder=4, alpha=0.7)
    
    # Draw trade-off nodes (orange squares)
    for node in tradeoff_features:
        if node in pos:
            ax.scatter(pos[node][0], pos[node][1], s=650, 
                      c='#ff7f0e', marker='s', edgecolors='black', linewidth=1, 
                      zorder=4, alpha=0.7)
    
    # Add labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=5, weight='bold', zorder=10)
    
    # Add legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='D', color='w', markerfacecolor='#1f77b4', 
              markersize=8, label='Selected Feature', markeredgecolor='black', markeredgewidth=1),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', 
              markersize=8, label='Alternative', markeredgecolor='black', markeredgewidth=1),
        mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', 
              markersize=8, label='Trade-off', markeredgecolor='black', markeredgewidth=1),
        # mlines.Line2D([0], [0], color='#1f77b4', linewidth=2, label='Correlation'),
        mlines.Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Alternative Link'),
        mlines.Line2D([0], [0], color='#ff7f0e', linewidth=2, linestyle=':', label='Trade-off Link'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)
    
    ax.set_xlim(min(x for x, y in pos.values()) - 0.2, max(x for x, y in pos.values()) + 0.2)
    ax.set_ylim(min(y for x, y in pos.values()) - 0.2, max(y for x, y in pos.values()) + 0.2)
    ax.axis('off')
    ax.set_title('Feature Correlation Network', fontsize=10, weight='bold', pad=10)

# ============================================================================
# VISUALIZATION: PER-CLUSTER DASHBOARDS (Comprehensive Cluster Highlights)
# ============================================================================



def plot_per_cluster_dashboards(df_clustered, insights, df_model_eval=None, 
                                output_dir='results/dashboards', use_high_shap_only=True):
    """
    Create comprehensive per-cluster dashboards with:
    - Top-left: Representative metric radar chart
    - Top-right: Trade-off analysis (representative metrics vs their trade-offs)
    - Bottom-left: Parallel coordinates (hyperparameters → metrics)
    - Bottom-right: Cluster statistics, rules, and key insights
    
    Each cluster gets its own PNG file for easy sharing/inclusion in reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_clusters = df_clustered['cluster'].max() + 1
    
    # Identify hyperparameter columns
    config_cols = ['criterion', 'fairness method', 'max depth', 'n estimators', 
                   'normalization', 'random state']
    config_cols = [col for col in config_cols if col in df_clustered.columns]
    
    colors = sns.color_palette("husl", n_clusters)
    
    for target_cluster_id in range(n_clusters):
        # Create dashboard figure with flexible layout
        fig = plt.figure(figsize=(18, 14))
        
        # Create a more flexible grid: allow different sizes per row
        # Row 0: Title (full width)
        # Row 1: Summary + HIGH Radar + LOW Radar (each 1 unit)
        # Row 2: Alternatives + Trade-offs (can be larger)
        # Row 3: Parallel Coords (full width)
        gs = fig.add_gridspec(4, 4, left=0.06, right=0.96, top=0.96, bottom=0.06,
                             hspace=0.4, wspace=0.35, height_ratios=[0.05, 0.35, 0.3, 0.4],
                             width_ratios=[1, 1.2, 1.2, 1])
        
        cluster_data = df_clustered[df_clustered['cluster'] == target_cluster_id]
        cluster_color = colors[target_cluster_id]
        
        # ========== ROW 0: TITLE ==========
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        # Load cluster insights early for summary
        cluster_insights = insights.get(str(target_cluster_id), {})
        summary = extract_cluster_metrics_summary(df_clustered, target_cluster_id, insights, n_top_metrics=3)
        
        # Extract SHAP data and alternatives early (needed for summary)
        shap_data = extract_shap_features_and_tradeoffs(df_clustered, target_cluster_id, insights, 
                                                        use_high_shap_only=use_high_shap_only)
        all_features = shap_data['all_features']
        values = shap_data['values']
        categories = shap_data['categories']
        high_features = shap_data['high_features']
        low_features = shap_data['low_features']
        
        # Extract alternatives and trade-offs (needed for summary)
        high_alternatives = extract_alternatives_and_tradeoffs(df_clustered, cluster_insights, shap_data, 'HIGH', all_features)
        low_alternatives = extract_alternatives_and_tradeoffs(df_clustered, cluster_insights, shap_data, 'LOW', all_features)
        
        # ========== ROW 1: CLUSTER SUMMARY + HIGH RADAR + LOW RADAR ==========
        # Summary on left (column 0)
        ax_summary = fig.add_subplot(gs[1, 0])
        ax_summary.axis('off')
        
        # Extract summary information
        n_alternatives = len(high_alternatives) + len(low_alternatives)
        n_tradeoffs = len([x for x in high_alternatives + low_alternatives if x['type'] == 'Trade-off'])
        
        # Get top decision rule
        decision_rules = cluster_insights.get('decision_tree_rules', [])
        top_rule = decision_rules[0]['rule'] if decision_rules else 'N/A'
        
        # Get medoid workflow ID
        medoid_id = cluster_insights.get('metadata', {}).get('medoid_workflow_id', 'N/A')
        
        # Get selected features - respect the use_high_shap_only setting
        if use_high_shap_only:
            # Use only high SHAP features
            high_shap_info = cluster_insights.get('high_shap_features', {})
            if isinstance(high_shap_info, dict):
                selected_feats = high_shap_info.get('features', [])
            else:
                selected_feats = []
        else:
            # Use all selected features from feature_selection
            selected_feats = cluster_insights.get('feature_selection', {}).get('selected_features', [])
        
        # Get quality interpretation
        quality_interp = cluster_insights.get('model_evaluation', {}).get('quality_interpretation', 'N/A')
        
        stats_text = "CLUSTER SUMMARY\n"
        stats_text += "─────────────────\n"
        stats_text += f"Size: {len(cluster_data)}\n"
        stats_text += f"%: {len(cluster_data)/len(df_clustered)*100:.1f}%\n"
        stats_text += f"Quality: {quality_interp}\n\n"
        stats_text += f"Selected Features: {len(selected_feats)}\n"
        for feat in selected_feats:
            stats_text += f"  • {feat}\n"
        
        stats_text += f"\nAlternatives: {len(high_alternatives)}\n"
        stats_text += f"Trade-offs: {n_tradeoffs}\n"
        stats_text += f"\nTop Rule:\n{top_rule}\n\n"
        stats_text += f"Medoid:\n{medoid_id}\n"
        
        ax_summary.text(0.05, 0.95, stats_text, transform=ax_summary.transAxes,
                       fontsize=6, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # HIGH RADAR (columns 1-2, larger)
        ax_high = fig.add_subplot(gs[1, 1:3], projection='polar')
        
        # Build HIGH metrics radar data - include selected features + alternatives
        high_plot_data = []
        
        # Add high selected features
        for feat in high_features:
            if feat in values:
                high_plot_data.append({
                    'name': feat,
                    'value': values[feat],
                    'type': 'Selected (HIGH)',
                    'category': 'HIGH'
                })
        
        # Add alternatives to high features
        high_alternatives_only = [x for x in high_alternatives if x['type'] == 'Alternative']
        for alt in high_alternatives_only:
            normalized_val = get_normalized_feature_value(df_clustered, target_cluster_id, alt['feature'])
            high_plot_data.append({
                'name': alt['feature'],
                'value': normalized_val,
                'type': 'Alternative',
                'related_to': alt['related_to']
            })
        
        other_cluster_ids = [i for i in range(n_clusters) if i != target_cluster_id][:2]
        
        if high_plot_data:
            N = len(high_plot_data)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles_plot = angles + angles[:1]
            
            # Plot all other clusters for context (greyed out background)
            n_clusters = df_clustered['cluster'].max() + 1
            all_other_cluster_ids = [i for i in range(n_clusters) if i != target_cluster_id]
            
            for other_id in all_other_cluster_ids:
                other_data = extract_shap_features_and_tradeoffs(df_clustered, other_id, insights,
                                                                 use_high_shap_only=use_high_shap_only)
                other_categories = other_data['categories']
                other_values = other_data['values']
                
                other_values_list = []
                for item in high_plot_data:
                    feat = item['name']
                    if other_categories.get(feat) == 'HIGH' and feat in other_values:
                        other_values_list.append(other_values[feat])
                    else:
                        other_values_list.append(0.5)
                
                other_values_plot = other_values_list + other_values_list[:1]
                ax_high.plot(angles_plot, other_values_plot, 'o-', linewidth=1.0, 
                            color='gray', alpha=0.15, markersize=2.5)
            
            # Plot target cluster
            target_values = [item['value'] for item in high_plot_data]
            target_values_plot = target_values + target_values[:1]
            
            ax_high.plot(angles_plot, target_values_plot, 'o-', linewidth=3.5, 
                         color=cluster_color, markersize=8, zorder=10)
            ax_high.fill(angles_plot, target_values_plot, alpha=0.25, color=cluster_color)
            
            # Labels with type indicators
            labels_high = []
            for item in high_plot_data:
                if item['type'] == 'Selected (HIGH)':
                    indicator = "^ "
                elif item['type'] == 'Alternative':
                    indicator = "◇ "
                else:  # Trade-off
                    indicator = "✕ "
                label = f"{indicator}{item['name']}"
                labels_high.append(label)
            
            ax_high.set_xticks(angles)
            ax_high.set_xticklabels(labels_high, size=7, wrap=True)
            ax_high.set_ylim(0, 1)
            ax_high.set_yticks([0.25, 0.5, 0.75])
            ax_high.set_yticklabels(['0.25', '0.5', '0.75'], size=7)
            ax_high.grid(True, linestyle='--', alpha=0.6)
            ax_high.set_title('HIGH Metrics', size=11, weight='bold', pad=15)
            
            # Add legend box on the right side of chart
            legend_text = "^ Selected HIGH\n◇ Alternative"
            # ax_high.text(1.35, 0.5, legend_text, transform=ax_high.transAxes,
            #             ha='left', va='center', fontsize=7, family='sans-serif',
            #             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.95))
        else:
            ax_high.text(0.5, 0.5, 'No HIGH features', ha='center', va='center',
                        transform=ax_high.transAxes, fontsize=11)
        
        # ========== ROW 1 RIGHT: LOW METRICS RADAR ==========
        ax_low = fig.add_subplot(gs[1, 3], projection='polar')
        
        # Build LOW metrics radar data - include selected features + alternatives
        low_plot_data = []
        
        # Add low selected features
        for feat in low_features:
            if feat in values:
                low_plot_data.append({
                    'name': feat,
                    'value': values[feat],
                    'type': 'Selected (LOW)',
                    'category': 'LOW'
                })
        
        # Add alternatives to low features
        low_alternatives_only = [x for x in low_alternatives if x['type'] == 'Alternative']
        for alt in low_alternatives_only:
            normalized_val = get_normalized_feature_value(df_clustered, target_cluster_id, alt['feature'])
            low_plot_data.append({
                'name': alt['feature'],
                'value': normalized_val,
                'type': 'Alternative',
                'related_to': alt['related_to']
            })
        
        if low_plot_data:
            N = len(low_plot_data)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles_plot = angles + angles[:1]
            
            # Plot all other clusters for context (greyed out background)
            n_clusters = df_clustered['cluster'].max() + 1
            all_other_cluster_ids = [i for i in range(n_clusters) if i != target_cluster_id]
            
            for other_id in all_other_cluster_ids:
                other_data = extract_shap_features_and_tradeoffs(df_clustered, other_id, insights,
                                                                 use_high_shap_only=use_high_shap_only)
                other_categories = other_data['categories']
                other_values = other_data['values']
                
                other_values_list = []
                for item in low_plot_data:
                    feat = item['name']
                    if other_categories.get(feat) == 'LOW' and feat in other_values:
                        other_values_list.append(other_values[feat])
                    else:
                        other_values_list.append(0.5)
                
                other_values_plot = other_values_list + other_values_list[:1]
                ax_low.plot(angles_plot, other_values_plot, 'o-', linewidth=1.0, 
                           color='gray', alpha=0.15, markersize=2.5)
            
            # Plot target cluster
            target_values = [item['value'] for item in low_plot_data]
            target_values_plot = target_values + target_values[:1]
            
            ax_low.plot(angles_plot, target_values_plot, 'o-', linewidth=3.5, 
                        color=cluster_color, markersize=8, zorder=10)
            ax_low.fill(angles_plot, target_values_plot, alpha=0.25, color=cluster_color)
            
            # Labels with type indicators
            labels_low = []
            for item in low_plot_data:
                if item['type'] == 'Selected (LOW)':
                    indicator = "v "
                elif item['type'] == 'Alternative':
                    indicator = "◇ "
                else:  # Trade-off
                    indicator = "✕ "
                label = f"{indicator}{item['name']}"
                labels_low.append(label)
            
            ax_low.set_xticks(angles)
            ax_low.set_xticklabels(labels_low, size=7, wrap=True)
            ax_low.set_ylim(0, 1)
            ax_low.set_yticks([0.25, 0.5, 0.75])
            ax_low.set_yticklabels(['0.25', '0.5', '0.75'], size=7)
            ax_low.grid(True, linestyle='--', alpha=0.6)
            ax_low.set_title('LOW Metrics', size=11, weight='bold', pad=15)
            
            # Add legend box on the right side of chart
            legend_text = "^ Selected LOW\n◇ Alternative"
            # ax_low.text(1.35, 0.5, legend_text, transform=ax_low.transAxes,
            #            ha='left', va='center', fontsize=7, family='sans-serif',
            #            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.95))
        else:
            ax_low.text(0.5, 0.5, 'No LOW features', ha='center', va='center',
                       transform=ax_low.transAxes, fontsize=11)
        
        # ========== ROW 2: FEATURE CORRELATION NETWORK ==========
        # Full width network graph (columns 0-3)
        ax_network = fig.add_subplot(gs[2, 0:4])
        
        plot_feature_correlation_network(ax_network, df_clustered, target_cluster_id, 
                                        cluster_insights, shap_data, high_alternatives, 
                                        low_alternatives, cluster_color)
        
        # ========== ROW 3: PARALLEL COORDINATES (full width) ==========
        ax_parallel = fig.add_subplot(gs[3, :])
        
        # Extract top metrics for parallel coordinates
        summary = extract_cluster_metrics_summary(df_clustered, target_cluster_id, insights, 
                                                  n_top_metrics=6)
        metric_cols_pc = summary['metrics'][:3]  # Use top 3 metrics
        
        all_cols_pc = config_cols + metric_cols_pc
        
        # Prepare data for both target cluster and all other clusters
        data_norm = pd.DataFrame()
        
        for col in config_cols:
            if col in cluster_data.columns:
                if cluster_data[col].dtype == 'object':
                    unique_vals = df_clustered[col].unique()
                    val_to_int = {v: i for i, v in enumerate(unique_vals)}
                    encoded = cluster_data[col].map(val_to_int)
                    col_min, col_max = 0, len(unique_vals) - 1
                else:
                    encoded = cluster_data[col]
                    col_min = df_clustered[col].min()
                    col_max = df_clustered[col].max()
                
                data_norm[col] = (encoded - col_min) / (col_max - col_min + 1e-8)
        
        for col in metric_cols_pc:
            col_min = df_clustered[col].min()
            col_max = df_clustered[col].max()
            data_norm[col] = (cluster_data[col] - col_min) / (col_max - col_min + 1e-8)
        
        # Plot all other clusters' data in grey background
        n_clusters_pc = df_clustered['cluster'].max() + 1
        for other_cluster_id in range(n_clusters_pc):
            if other_cluster_id != target_cluster_id:
                other_cluster_data = df_clustered[df_clustered['cluster'] == other_cluster_id]
                
                # Normalize other cluster data using same scaling as target
                other_data_norm = pd.DataFrame()
                for col in config_cols:
                    if col in other_cluster_data.columns:
                        if other_cluster_data[col].dtype == 'object':
                            unique_vals = df_clustered[col].unique()
                            val_to_int = {v: i for i, v in enumerate(unique_vals)}
                            encoded = other_cluster_data[col].map(val_to_int)
                            col_min, col_max = 0, len(unique_vals) - 1
                        else:
                            encoded = other_cluster_data[col]
                            col_min = df_clustered[col].min()
                            col_max = df_clustered[col].max()
                        other_data_norm[col] = (encoded - col_min) / (col_max - col_min + 1e-8)
                
                for col in metric_cols_pc:
                    col_min = df_clustered[col].min()
                    col_max = df_clustered[col].max()
                    other_data_norm[col] = (other_cluster_data[col] - col_min) / (col_max - col_min + 1e-8)
                
                # Plot other cluster lines in background
                for idx, row in other_data_norm.iterrows():
                    ax_parallel.plot(range(len(all_cols_pc)), row.values, 
                                   color='gray', alpha=0.05, linewidth=0.5, zorder=1)
        
        # Plot target cluster lines (on top)
        for idx, row in data_norm.iterrows():
            ax_parallel.plot(range(len(all_cols_pc)), row.values, 
                           color=cluster_color, alpha=0.1, linewidth=0.7, zorder=10)
        
        # Plot mean line
        mean_line = data_norm.mean()
        ax_parallel.plot(range(len(all_cols_pc)), mean_line.values, 
                        color=cluster_color, alpha=1.0, linewidth=3, 
                        marker='o', markersize=6, label='Mean', zorder=100)
        
        ax_parallel.set_xticks(range(len(all_cols_pc)))
        labels_pc = []
        for i, col in enumerate(all_cols_pc):
            if i == len(config_cols):
                labels_pc.append('METRICS →\n' + col)
            else:
                labels_pc.append(col)
        
        ax_parallel.set_xticklabels(labels_pc, rotation=45, ha='right', fontsize=7, wrap=True)
        ax_parallel.set_ylim(-0.05, 1.05)
        ax_parallel.set_ylabel('Normalized Value', fontsize=10)
        ax_parallel.axvline(x=len(config_cols) - 0.5, color='black', linestyle=':', 
                           linewidth=1.5, alpha=0.5)
        ax_parallel.grid(True, axis='y', alpha=0.2)
        ax_parallel.set_title('Configuration → Metrics\n(Parallel Coordinates)', 
                            size=11, weight='bold')
        ax_parallel.legend(fontsize=9, loc='upper right')
        
        # Main title
        fig.suptitle(f'Cluster {target_cluster_id} Analysis Dashboard', 
                    fontsize=16, weight='bold', y=0.98)
        
        # Save
        output_file = os.path.join(output_dir, f'cluster_{target_cluster_id}_dashboard.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved cluster {target_cluster_id} dashboard to {output_file}")
        plt.close()
    
    print(f"\n✓ All {n_clusters} cluster dashboards saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations for cluster highlights."""
    print("="*80)
    print("WORKFLOW CLUSTER HIGHLIGHTS VISUALIZATION")
    print("="*80 + "\n")
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python 3_view_results.py <data_folder>")
        print("\nArguments:")
        print("  data_folder - Path to folder containing clustering results (e.g., data/workflows or data/workflows_big)")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    
    # Load data
    print("Loading cluster data and insights...")
    df_clustered, insights, df_model_eval = load_cluster_data(data_folder)
    print(f"✓ Loaded {len(df_clustered)} workflows in {df_clustered['cluster'].max() + 1} clusters\n")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate all visualizations
    print("Generating visualizations...\n")
    
    # Configuration: Set to True for high SHAP features only, False for all selected features
    USE_HIGH_SHAP_ONLY = False
    feature_mode = "High SHAP Features Only" if USE_HIGH_SHAP_ONLY else "All Selected Features"
    print(f"Representative Feature Mode: {feature_mode}\n")
    
    print("1. Per-Cluster Dashboards (Radar + Trade-offs + Parallel Coords + Stats)")
    plot_per_cluster_dashboards(df_clustered, insights, df_model_eval, 'results/dashboards',
                               use_high_shap_only=USE_HIGH_SHAP_ONLY)
    
    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print("Output directory: results/")
    print("="*80)


if __name__ == "__main__":
    main()