"""
Cluster Ranking & Selection Pipeline
=====================================
Implements cluster-adaptive ranking system to:
1. MACRO-RANKING (Global): Identify efficient (non-dominated) clusters using Pareto dominance
2. CONSENSUS-RANKING: Rank all clusters using Borda count and Copeland's method
3. MICRO-RANKING (Local): Rank runs within each cluster based on benefit-to-cost ratio

The ranking considers:
- Representative Metrics (R_k): Primary objectives from SHAP analysis
- Trade-offs: Negatively correlated features (r <= -0.75) representing costs/constraints

References methodology from ranking_ideas.md
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def load_cluster_data(data_folder='data/workflows'):
    """Load all clustering and insights data similar to 3_view_results.py"""
    df_clustered = pd.read_csv(os.path.join(data_folder, "workflows_clustered.csv"))
    
    with open(os.path.join(data_folder, "clusters_comprehensive_insights.json"), 'r') as f:
        insights = json.load(f)
    
    try:
        df_model_eval = pd.read_csv(os.path.join(data_folder, "workflows_model_evaluation_summary.csv"))
    except FileNotFoundError:
        df_model_eval = None
    
    try:
        df_medoids = pd.read_csv(os.path.join(data_folder, "workflows_medoids.csv"))
    except FileNotFoundError:
        df_medoids = None
    
    return df_clustered, insights, df_model_eval, df_medoids


def extract_cluster_signature(insights, cluster_id, correlation_threshold=0.75):
    """
    Extract the Cluster Signature (R_k ∪ Trade-offs) for a cluster.
    
    Trade-offs are metric-specific: benefit metric paired with cost metric.
    Only includes representatives with "high" or "low" value_category (excludes "mid").
    
    Returns:
        Dictionary with:
        - representatives: Primary metrics from SHAP analysis (benefits) - only high/low
        - tradeoff_pairs: List of (benefit, cost) tuples with correlation strength
        - all_metrics: Union of representatives and cost metrics
    """
    cluster_insights = insights.get(str(cluster_id), {})
    
    # Extract representatives from high SHAP features
    # Only include features with "high" or "low" value_category (skip "mid")
    representatives = []
    representative_value_categories = {}
    high_shap_info = cluster_insights.get('high_shap_features', {})
    if isinstance(high_shap_info, dict):
        high_shap_features = high_shap_info.get('features', [])
        feature_stats = high_shap_info.get('feature_statistics', {})
        
        if isinstance(high_shap_features, str):
            candidate_features = [f.strip() for f in high_shap_features.split(',') if f.strip()]
        elif isinstance(high_shap_features, list):
            candidate_features = [str(f).strip() for f in high_shap_features if f]
        else:
            candidate_features = []
        
        # Filter: only keep features with "high" or "low" value_category
        for feat in candidate_features:
            feat_info = feature_stats.get(feat, {})
            value_category = feat_info.get('value_category', 'mid')
            if value_category in ['high', 'low']:
                representatives.append(feat)
                representative_value_categories[feat] = value_category
    
    # Extract trade-offs as (benefit, cost) pairs from trade_off_analysis
    # Trade-off pairs are benefit-cost relationships (negative correlation)
    # IMPORTANT: A valid trade-off MUST include at least one representative metric
    tradeoff_pairs = []  # List of {'benefit': metric, 'cost': metric, 'correlation': float}
    tradeoff_analysis = cluster_insights.get('trade_off_analysis', {})
    strong_tradeoffs = tradeoff_analysis.get('strong_tradeoffs', [])
    
    for trade in strong_tradeoffs:
        m1 = trade.get('metric_1', '')
        m2 = trade.get('metric_2', '')
        corr = trade.get('relationship_strength', 0)
        is_tradeoff = trade.get('is_tradeoff', 0)
        
        # Only process actual trade-offs (negative correlation)
        if (is_tradeoff == 1):
            # Determine which is benefit (representative) and which is cost
            m1_is_rep = m1 in representatives
            m2_is_rep = m2 in representatives
            
            # A valid trade-off MUST include at least one representative
            if m1_is_rep or m2_is_rep:
                if m1_is_rep and not m2_is_rep:
                    # m1 is benefit (representative), m2 is cost
                    tradeoff_pairs.append({
                        'benefit': m1,
                        'cost': m2,
                        'correlation': corr
                    })
                elif m2_is_rep and not m1_is_rep:
                    # m2 is benefit (representative), m1 is cost
                    tradeoff_pairs.append({
                        'benefit': m2,
                        'cost': m1,
                        'correlation': corr
                    })
                else:
                    # Both are representatives - trade-off between two benefits
                    # Use m1 as benefit, m2 as cost (arbitrary but consistent)
                    tradeoff_pairs.append({
                        'benefit': m1,
                        'cost': m2,
                        'correlation': corr
                    })
            # Skip trade-offs where neither metric is a representative
    
    # Collect all metrics for normalization
    all_cost_metrics = [p['cost'] for p in tradeoff_pairs]
    all_metrics = list(set(representatives + all_cost_metrics))
    
    return {
        'representatives': representatives,
        'representative_value_categories': representative_value_categories,
        'tradeoff_pairs': tradeoff_pairs,
        'all_metrics': all_metrics,
        'n_representatives': len(representatives),
        'n_tradeoff_pairs': len(tradeoff_pairs)
    }


def get_metric_directionality(df_clustered, metric):
    """
    Determine if higher or lower values are better for a metric.
    Uses heuristics based on metric names.
    
    Returns: 1 if 'higher is better', -1 if 'lower is better'
    """
    metric_lower = metric.lower()
    
    # Higher is better
    if any(keyword in metric_lower for keyword in 
           ['accuracy', 'auc', 'f1', 'precision', 'recall', 'score', 'performance', 'roc']):
        return 1
    
    # Lower is better
    if any(keyword in metric_lower for keyword in 
           ['loss', 'error', 'latency', 'time', 'cost', 'memory', 'cpu', 'disk']):
        return -1
    
    # Default: higher is better
    return 1


def compute_hero_run_for_cluster(signature, cluster_data, df_clustered):
    """
    Compute a "hero" run within a cluster (best trade-off) using the same logic
    as the micro-ranking step, but without requiring Pareto efficiency.

    Returns a dict with:
      - row (Series) for the chosen run
      - workflow_id
      - score components
    """
    if cluster_data.empty:
        return None

    all_metrics = signature['all_metrics']
    if not all_metrics:
        return None

    # Build ideal point per metric
    ideal_point = {}
    for metric in all_metrics:
        if metric in df_clustered.columns:
            direction = get_metric_directionality(df_clustered, metric)
            if direction == 1:
                ideal_point[metric] = cluster_data[metric].max()
            else:
                ideal_point[metric] = cluster_data[metric].min()

    # Normalize metrics to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    metrics_data = cluster_data.reindex(columns=all_metrics).copy()
    for metric in all_metrics:
        if metric not in metrics_data.columns:
            metrics_data[metric] = 0.5

    metrics_normalized = pd.DataFrame(
        scaler.fit_transform(metrics_data.fillna(metrics_data.mean())),
        columns=all_metrics,
        index=metrics_data.index
    )

    run_scores = []
    alpha, gamma = 0.70, 0.30

    for run_idx in cluster_data.index:
        score_components = {'representatives': 0.0, 'tradeoffs': 0.0}

        # Representatives distance
        rep_distances = []
        for metric in signature['representatives']:
            if metric in metrics_normalized.columns:
                actual_val = metrics_normalized.loc[run_idx, metric]
                direction = get_metric_directionality(df_clustered, metric)
                distance = abs(1.0 - actual_val) if direction == 1 else abs(actual_val)
                rep_distances.append(distance)
        score_components['representatives'] = np.mean(rep_distances) if rep_distances else 0.0

        # Trade-off costs distance
        trade_distances = []
        for tradeoff in signature['tradeoff_pairs']:
            cost_metric = tradeoff['cost']
            if cost_metric in metrics_normalized.columns:
                actual_val = metrics_normalized.loc[run_idx, cost_metric]
                direction = get_metric_directionality(df_clustered, cost_metric)
                distance = abs(actual_val) if direction == 1 else abs(0 - actual_val)
                trade_distances.append(distance)
        score_components['tradeoffs'] = np.mean(trade_distances) if trade_distances else 0.0

        ensemble_score = alpha * score_components['representatives'] + gamma * score_components['tradeoffs']

        run_scores.append({
            'run_idx': run_idx,
            'workflow_id': cluster_data.loc[run_idx, 'workflowId'] if 'workflowId' in cluster_data.columns else str(run_idx),
            'representatives_dist': score_components['representatives'],
            'tradeoffs_dist': score_components['tradeoffs'],
            'ensemble_score': ensemble_score
        })

    run_scores.sort(key=lambda x: x['ensemble_score'])
    best = run_scores[0]

    return {
        'row': cluster_data.loc[best['run_idx']],
        'workflow_id': best['workflow_id'],
        'ensemble_score': best['ensemble_score'],
        'representatives_dist': best['representatives_dist'],
        'tradeoffs_dist': best['tradeoffs_dist']
    }


def select_cluster_metric_row(cluster_id, cluster_data, df_clustered, df_medoids, signature, metric_source='mean'):
    """
    Select the row/values used to represent a cluster when comparing metrics.

    metric_source options:
      - 'mean': use cluster mean (legacy behavior)
      - 'medoid': use cluster medoid row if available
      - 'hero': use best (hero) run within the cluster
    """
    metric_source = metric_source.lower()

    if metric_source == 'mean':
        return cluster_data.mean(numeric_only=True)

    if metric_source == 'medoid' and df_medoids is not None:
        medoid_row = df_medoids[df_medoids['cluster_id'] == cluster_id]
        if not medoid_row.empty:
            medoid_index = medoid_row.iloc[0]['medoid_index']
            try:
                medoid_index = int(medoid_index)
            except Exception:
                pass
            if medoid_index in cluster_data.index:
                return cluster_data.loc[medoid_index]
            # Fallback via workflow_id match if index not aligned
            workflow_id = medoid_row.iloc[0].get('workflow_id')
            if workflow_id is not None and 'workflowId' in cluster_data.columns:
                matched = cluster_data[cluster_data['workflowId'] == workflow_id]
                if not matched.empty:
                    return matched.iloc[0]
        # Fallback to mean if medoid missing
        return cluster_data.mean(numeric_only=True)

    if metric_source == 'hero':
        hero = compute_hero_run_for_cluster(signature, cluster_data, df_clustered)
        if hero:
            return hero['row']
        return cluster_data.mean(numeric_only=True)

    # Default fallback
    return cluster_data.mean(numeric_only=True)


def analyze_cluster_dominance(cluster_signatures, cluster_metrics, n_clusters, df_clustered, dominated_by_macro):
    """
    Analyze and document where each cluster dominates others and how.
    
    Uses the dominance results from macro_ranking to ensure consistency.
    
    Returns a detailed analysis of:
    - Which clusters each cluster dominates
    - On which metrics each cluster excels
    - The magnitude of dominance (percentage improvement)
    """
    print("\n" + "-"*80)
    print("DETAILED CLUSTER DOMINANCE ANALYSIS")
    print("-"*80)
    
    dominance_report = {}
    
    for cluster_id in range(n_clusters):
        sig = cluster_signatures[cluster_id]
        metrics_this = cluster_metrics[cluster_id]
        rep_value_cats = sig.get('representative_value_categories', {})
        
        dominance_report[cluster_id] = {
            'representatives': sig['representatives'],
            'tradeoff_pairs': sig['tradeoff_pairs'],
            'dominates_others': [],
            'dominated_by': [],
            'strengths': [],
            'weaknesses': []
        }
        
        # Collect all metrics where this cluster excels
        # For each of this cluster's representative metrics
        for metric in sig['representatives']:
            if metric not in metrics_this:
                continue
            
            direction = get_metric_directionality(df_clustered, metric)
            value_category = rep_value_cats.get(metric, 'high')
            aligned_preference = (
                (value_category == 'high' and direction == 1) or
                (value_category == 'low' and direction == -1)
            )
            this_val = metrics_this.get(metric, 0)
            
            # Compare with other clusters
            better_count = 0
            worse_count = 0
            not_their_rep_count = 0  # Clusters where this metric is not their representative
            improvements = []
            
            for other_id in range(n_clusters):
                if other_id == cluster_id:
                    continue
                
                other_sig = cluster_signatures[other_id]
                other_val = cluster_metrics[other_id].get(metric, None)
                
                # Check if this metric is a representative for the other cluster
                metric_is_other_rep = metric in other_sig['representatives']
                
                if not metric_is_other_rep:
                    # They don't even focus on this metric - we beat them by default
                    not_their_rep_count += 1
                    better_count += 1
                    if other_val is not None and other_val != 0:
                        if direction == 1:
                            pct_improvement = ((this_val - other_val) / abs(other_val)) * 100
                        else:
                            pct_improvement = ((other_val - this_val) / abs(other_val)) * 100
                    else:
                        pct_improvement = 0
                    improvements.append((other_id, pct_improvement, 'not_their_focus'))
                elif other_val is not None:
                    # They also focus on this metric - direct competition
                    if direction == 1:  # Higher is better
                        if this_val > other_val:
                            better_count += 1
                            if other_val != 0:
                                pct_improvement = ((this_val - other_val) / abs(other_val)) * 100
                            else:
                                pct_improvement = 100 if this_val > 0 else 0
                            improvements.append((other_id, pct_improvement, 'direct_competition'))
                        elif this_val < other_val:
                            worse_count += 1
                    else:  # Lower is better
                        if this_val < other_val:
                            better_count += 1
                            if other_val != 0:
                                pct_improvement = ((other_val - this_val) / abs(other_val)) * 100
                            else:
                                pct_improvement = 100 if this_val < 0 else 0
                            improvements.append((other_id, pct_improvement, 'direct_competition'))
                        elif this_val > other_val:
                            worse_count += 1
            
            # Classify as strength or weakness, incorporating value_category preference
            if not aligned_preference:
                dominance_report[cluster_id]['weaknesses'].append({
                    'metric': metric,
                    'value': this_val,
                    'direction': 'higher' if direction == 1 else 'lower',
                    'value_category': value_category,
                    'worse_than_n_clusters': worse_count,
                    'reason': 'value_category_mismatch'
                })
            elif better_count > worse_count:
                avg_improvement = np.mean([imp[1] for imp in improvements]) if improvements else 0
                dominance_report[cluster_id]['strengths'].append({
                    'metric': metric,
                    'value': this_val,
                    'direction': 'higher' if direction == 1 else 'lower',
                    'value_category': value_category,
                    'beats_n_clusters': better_count,
                    'beats_by_default': not_their_rep_count,  # Not their representative
                    'beats_in_competition': better_count - not_their_rep_count,  # Direct wins
                    'avg_improvement_pct': avg_improvement,
                    'improvements_over': improvements
                })
            elif worse_count > better_count:
                dominance_report[cluster_id]['weaknesses'].append({
                    'metric': metric,
                    'value': this_val,
                    'direction': 'higher' if direction == 1 else 'lower',
                    'value_category': value_category,
                    'worse_than_n_clusters': worse_count
                })
    
    # Use the dominance results from macro-ranking to ensure consistency
    # Build reverse mapping: who dominates whom
    dominates_map = {cid: set() for cid in range(n_clusters)}
    for dominated_cluster, dominating_clusters in dominated_by_macro.items():
        for dominator in dominating_clusters:
            dominates_map[dominator].add(dominated_cluster)
    
    # Now build detailed explanations for the dominance relationships
    for i in range(n_clusters):
        for j in dominates_map[i]:  # Only process clusters that i actually dominates
            sig_j = cluster_signatures[j]
            metrics_i = cluster_metrics[i]
            metrics_j = cluster_metrics[j]
            
            dominance_reasons = []
            
            # Explain why i dominates j on j's representatives
            for metric in sig_j['representatives']:
                if metric in metrics_i and metric in metrics_j:
                    direction = get_metric_directionality(df_clustered, metric)
                    val_i = metrics_i[metric]
                    val_j = metrics_j[metric]
                    
                    if val_j != 0:
                        if direction == 1:
                            pct = ((val_i - val_j) / abs(val_j)) * 100
                        else:
                            pct = ((val_j - val_i) / abs(val_j)) * 100
                    else:
                        pct = 0
                    
                    dominance_reasons.append({
                        'metric': metric,
                        'type': 'representative',
                        'cluster_i_value': val_i,
                        'cluster_j_value': val_j,
                        'improvement_pct': pct,
                        'direction': 'higher is better' if direction == 1 else 'lower is better'
                    })
            
            # Explain why i dominates j on j's trade-off costs
            for tradeoff_pair in sig_j['tradeoff_pairs']:
                cost_metric = tradeoff_pair['cost']
                if cost_metric in metrics_i and cost_metric in metrics_j:
                    direction = get_metric_directionality(df_clustered, cost_metric)
                    val_i = metrics_i[cost_metric]
                    val_j = metrics_j[cost_metric]
                    
                    if val_j != 0:
                        if direction == -1:  # Lower is better for costs
                            pct = ((val_j - val_i) / abs(val_j)) * 100
                        else:
                            pct = ((val_i - val_j) / abs(val_j)) * 100
                    else:
                        pct = 0
                    
                    dominance_reasons.append({
                        'metric': cost_metric,
                        'type': 'tradeoff_cost',
                        'related_benefit': tradeoff_pair['benefit'],
                        'cluster_i_value': val_i,
                        'cluster_j_value': val_j,
                        'improvement_pct': pct,
                        'direction': 'lower is better (cost)' if direction == -1 else 'higher is better'
                    })
            
            if dominance_reasons:
                dominance_report[i]['dominates_others'].append({
                    'dominated_cluster': j,
                    'reasons': dominance_reasons
                })
                dominance_report[j]['dominated_by'].append({
                    'dominating_cluster': i,
                    'reasons': dominance_reasons
                })
    
    # Print detailed dominance report
    for cluster_id in range(n_clusters):
        report = dominance_report[cluster_id]
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} DOMINANCE PROFILE")
        print(f"{'='*60}")
        
        # Representatives
        print(f"\n📊 Representative Metrics: {', '.join(report['representatives']) if report['representatives'] else 'None'}")
        
        # Trade-off pairs
        if report['tradeoff_pairs']:
            print(f"\n⚖️  Trade-off Pairs:")
            for tp in report['tradeoff_pairs']:
                print(f"    - {tp['benefit']} ↔ {tp['cost']} (r = {tp['correlation']:.2f})")
        
        # Strengths
        if report['strengths']:
            print(f"\n✅ STRENGTHS (Where Cluster {cluster_id} Excels on its Representatives):")
            for strength in sorted(report['strengths'], key=lambda x: -x['avg_improvement_pct']):
                direction_str = "↑ higher is better" if strength['direction'] == 'higher' else "↓ lower is better"
                print(f"    • {strength['metric']} = {strength['value']:.4f} ({direction_str}, prefers {strength.get('value_category', 'high')})")
                beats_default = strength.get('beats_by_default', 0)
                beats_competition = strength.get('beats_in_competition', 0)
                print(f"      Beats {strength['beats_n_clusters']}/{n_clusters-1} clusters:")
                print(f"        - {beats_default} by default (not their representative)")
                print(f"        - {beats_competition} in direct competition")
                print(f"      Avg improvement: {strength['avg_improvement_pct']:.1f}%")
        
        # Weaknesses
        if report['weaknesses']:
            print(f"\n❌ WEAKNESSES (Where Cluster {cluster_id} Lags on its Representatives):")
            for weakness in report['weaknesses']:
                direction_str = "↑ higher is better" if weakness['direction'] == 'higher' else "↓ lower is better"
                vc = weakness.get('value_category', 'high')
                reason = weakness.get('reason')
                if reason == 'value_category_mismatch':
                    print(f"    • {weakness['metric']} = {weakness['value']:.4f} ({direction_str}, prefers {vc})")
                    print(f"      Weakness due to preference mismatch (expected {vc})")
                else:
                    print(f"    • {weakness['metric']} = {weakness['value']:.4f} ({direction_str}, prefers {vc})")
                    print(f"      Worse than {weakness['worse_than_n_clusters']}/{n_clusters-1} clusters (who also have this as representative)")
        
        # Dominates which clusters
        if report['dominates_others']:
            print(f"\n🏆 DOMINATES OTHER CLUSTERS:")
            for dom in report['dominates_others']:
                print(f"\n    ▶ Cluster {cluster_id} DOMINATES Cluster {dom['dominated_cluster']}:")
                for reason in dom['reasons']:
                    metric_type = "📈 Representative" if reason['type'] == 'representative' else "💰 Trade-off Cost"
                    if reason['type'] == 'tradeoff_cost':
                        print(f"      {metric_type}: {reason['metric']} (cost for {reason['related_benefit']})")
                    else:
                        print(f"      {metric_type}: {reason['metric']}")
                    print(f"        Cluster {cluster_id}: {reason['cluster_i_value']:.4f} vs Cluster {dom['dominated_cluster']}: {reason['cluster_j_value']:.4f}")
                    print(f"        Improvement: {reason['improvement_pct']:.1f}% ({reason['direction']})")
        else:
            print(f"\n🏆 Does not dominate any cluster (may be non-dominated/efficient)")
        
        # Dominated by which clusters
        if report['dominated_by']:
            print(f"\n⬇️  DOMINATED BY:")
            for dom in report['dominated_by']:
                print(f"    Cluster {dom['dominating_cluster']} (on {len(dom['reasons'])} metric(s))")
        else:
            print(f"\n⬇️  Not dominated by any cluster (EFFICIENT / Pareto Optimal)")
    
    return dominance_report


def borda_ranking_clusters(cluster_signatures, cluster_metrics, n_clusters, df_clustered):
    """
    Compute Borda count ranking for all clusters based on representative metrics.
    
    For each representative metric across all clusters:
    1. Rank clusters from best to worst on that metric
    2. Assign Borda points: best gets (n_clusters - 1), worst gets 0
    3. Sum Borda points across all representative metrics
    
    This provides a consensus ranking that considers all representative metrics.
    
    Returns:
        DataFrame with cluster rankings and Borda scores
    """
    print("\n" + "="*80)
    print("BORDA COUNT RANKING (Consensus Ranking Across All Representatives)")
    print("="*80)
    
    # Collect all unique representative metrics across all clusters
    all_representatives = set()
    for cluster_id in range(n_clusters):
        sig = cluster_signatures[cluster_id]
        all_representatives.update(sig['representatives'])
    
    all_representatives = sorted(list(all_representatives))
    
    if not all_representatives:
        print("⚠ No representative metrics found for Borda ranking")
        return pd.DataFrame()
    
    print(f"\nRanking clusters on {len(all_representatives)} representative metrics:")
    print(f"  {', '.join(all_representatives)}")
    
    # Initialize Borda scores
    borda_scores = {cid: 0 for cid in range(n_clusters)}
    borda_details = {cid: {} for cid in range(n_clusters)}
    
    # For each representative metric, rank all clusters
    for metric in all_representatives:
        print(f"\n  Ranking on {metric}...")
        
        # Get metric values for all clusters
        metric_values = []
        for cluster_id in range(n_clusters):
            val = cluster_metrics[cluster_id].get(metric, None)
            if val is not None:
                metric_values.append((cluster_id, val))
        
        if not metric_values:
            continue
        
        # Determine ranking order based on metric directionality
        direction = get_metric_directionality(df_clustered, metric)
        reverse = (direction == 1)  # Higher is better -> reverse sort
        
        # Sort clusters by metric value
        metric_values.sort(key=lambda x: x[1], reverse=reverse)
        
        # Assign Borda points: best gets (n_clusters - 1), worst gets 0
        for rank_position, (cluster_id, value) in enumerate(metric_values):
            borda_points = len(metric_values) - 1 - rank_position
            borda_scores[cluster_id] += borda_points
            borda_details[cluster_id][metric] = {
                'rank': rank_position + 1,
                'value': value,
                'borda_points': borda_points
            }
            
            if rank_position < 3:  # Print top 3
                print(f"    #{rank_position + 1}: Cluster {cluster_id} = {value:.4f} ({borda_points} points)")
    
    # Create ranking dataframe
    ranking_data = []
    for cluster_id in range(n_clusters):
        sig = cluster_signatures[cluster_id]
        ranking_data.append({
            'cluster_id': cluster_id,
            'borda_score': borda_scores[cluster_id],
            'n_representatives': sig['n_representatives'],
            'n_metrics_ranked_on': len([m for m in all_representatives if m in borda_details[cluster_id]]),
            'cluster_size': len(df_clustered[df_clustered['cluster'] == cluster_id])
        })
    
    borda_df = pd.DataFrame(ranking_data)
    borda_df = borda_df.sort_values('borda_score', ascending=False)
    borda_df['borda_rank'] = range(1, len(borda_df) + 1)
    
    print("\n" + "-"*80)
    print("BORDA RANKING RESULTS")
    print("-"*80)
    print(f"\n{'Rank':<6} {'Cluster':<10} {'Borda Score':<15} {'Representatives':<18} {'Cluster Size'}")
    print("-"*80)
    for _, row in borda_df.iterrows():
        print(f"{row['borda_rank']:<6} {int(row['cluster_id']):<10} {int(row['borda_score']):<15} {int(row['n_representatives']):<18} {int(row['cluster_size'])}")
    
    return {
        'borda_ranking': borda_df,
        'borda_scores': borda_scores,
        'borda_details': borda_details,
        'all_representatives': all_representatives
    }


def copeland_ranking_clusters(cluster_signatures, cluster_metrics, n_clusters, df_clustered):
    """
    Compute Copeland count ranking for all clusters based on representative metrics.
    
    Copeland's method uses pairwise comparisons:
    1. For each pair of clusters (i, j)
    2. For each representative metric k:
       - If cluster i is better than j on metric k, i gets 1 point vs j
       - If j is better than i on metric k, j gets 1 point vs i
    3. Rank clusters by total pairwise wins
    
    This provides a consensus ranking based on head-to-head performance.
    
    Returns:
        DataFrame with cluster rankings and Copeland scores
    """
    print("\n" + "="*80)
    print("COPELAND'S RANKING (Pairwise Head-to-Head Comparison)")
    print("="*80)
    
    # Collect all unique representative metrics across all clusters
    all_representatives = set()
    for cluster_id in range(n_clusters):
        sig = cluster_signatures[cluster_id]
        all_representatives.update(sig['representatives'])
    
    all_representatives = sorted(list(all_representatives))
    
    if not all_representatives:
        print("⚠ No representative metrics found for Copeland ranking")
        return pd.DataFrame()
    
    print(f"\nComparing clusters on {len(all_representatives)} representative metrics:")
    print(f"  {', '.join(all_representatives)}")
    
    # Initialize Copeland scores (pairwise wins)
    copeland_scores = {cid: 0 for cid in range(n_clusters)}
    copeland_details = {cid: {} for cid in range(n_clusters)}
    
    # Pairwise comparisons
    print("\n" + "-"*80)
    print("Pairwise Comparisons")
    print("-"*80)
    
    for i, j in combinations(range(n_clusters), 2):
        i_wins = 0
        j_wins = 0
        comparison_details = []
        
        # Compare on each representative metric
        for metric in all_representatives:
            val_i = cluster_metrics[i].get(metric, None)
            val_j = cluster_metrics[j].get(metric, None)
            
            if val_i is not None and val_j is not None:
                direction = get_metric_directionality(df_clustered, metric)
                
                if direction == 1:  # Higher is better
                    if val_i > val_j:
                        i_wins += 1
                    elif val_j > val_i:
                        j_wins += 1
                else:  # Lower is better
                    if val_i < val_j:
                        i_wins += 1
                    elif val_j < val_i:
                        j_wins += 1
                
                comparison_details.append({
                    'metric': metric,
                    'val_i': val_i,
                    'val_j': val_j,
                    'winner': 'i' if i_wins > j_wins else ('j' if j_wins > i_wins else 'tie')
                })
        
        # Award points for this pairwise comparison
        if i_wins > j_wins:
            copeland_scores[i] += 1
            copeland_details[i][j] = {'wins': i_wins, 'losses': j_wins}
            copeland_details[j][i] = {'wins': j_wins, 'losses': i_wins}
        elif j_wins > i_wins:
            copeland_scores[j] += 1
            copeland_details[i][j] = {'wins': i_wins, 'losses': j_wins}
            copeland_details[j][i] = {'wins': j_wins, 'losses': i_wins}
        else:
            # Tie: both get 0.5 points (or could split differently)
            copeland_scores[i] += 0.5
            copeland_scores[j] += 0.5
            copeland_details[i][j] = {'wins': i_wins, 'losses': j_wins}
            copeland_details[j][i] = {'wins': j_wins, 'losses': i_wins}
    
    # Create ranking dataframe
    ranking_data = []
    for cluster_id in range(n_clusters):
        sig = cluster_signatures[cluster_id]
        ranking_data.append({
            'cluster_id': cluster_id,
            'copeland_score': copeland_scores[cluster_id],
            'n_representatives': sig['n_representatives'],
            'cluster_size': len(df_clustered[df_clustered['cluster'] == cluster_id])
        })
    
    copeland_df = pd.DataFrame(ranking_data)
    copeland_df = copeland_df.sort_values('copeland_score', ascending=False)
    copeland_df['copeland_rank'] = range(1, len(copeland_df) + 1)
    
    print("\n" + "-"*80)
    print("COPELAND'S RANKING RESULTS")
    print("-"*80)
    print(f"\n{'Rank':<6} {'Cluster':<10} {'Copeland Score':<18} {'Representatives':<18} {'Cluster Size'}")
    print("-"*80)
    for _, row in copeland_df.iterrows():
        print(f"{row['copeland_rank']:<6} {int(row['cluster_id']):<10} {row['copeland_score']:<18.1f} {int(row['n_representatives']):<18} {int(row['cluster_size'])}")
    
    return {
        'copeland_ranking': copeland_df, 
        'copeland_scores': copeland_scores,
        'copeland_details': copeland_details,
        'all_representatives': all_representatives
    }


def macro_ranking_pareto_dominance(df_clustered, insights, n_clusters, df_medoids=None, metric_source='mean'):
    """
    LEVEL 1: MACRO-RANKING (Global Efficiency)
    
    Identifies efficient clusters using Pareto dominance with trade-offs.
    Cluster A dominates Cluster B if:
    1. A is better/equal on B's representatives
    2. A is better/equal on B's trade-offs (minimizing costs)
    
    Returns:
        Dictionary with:
        - efficient_clusters: List of non-dominated cluster IDs
        - dominated_clusters: Dict mapping dominated cluster to dominating clusters
        - pareto_frontier: DataFrame with efficiency metrics
    """
    print("\n" + "="*80)
    print("LEVEL 1: MACRO-RANKING (Global Efficiency via Pareto Dominance)")
    print("="*80)
    
    cluster_signatures = {}
    cluster_metrics = {}
    
    # First pass: Extract all signatures to collect ALL metrics across all clusters
    all_metrics_global = set()
    for cluster_id in range(n_clusters):
        signature = extract_cluster_signature(insights, cluster_id)
        cluster_signatures[cluster_id] = signature
        all_metrics_global.update(signature['all_metrics'])
    
    # Second pass: Compute ALL metrics for ALL clusters (not just their own signature metrics)
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        metrics_values = {}

        # Select representative row based on configured metric source
        representative_row = select_cluster_metric_row(
            cluster_id, cluster_data, df_clustered, df_medoids, cluster_signatures[cluster_id], metric_source
        )

        # Collect values for ALL metrics that exist in any cluster's signature
        for metric in all_metrics_global:
            if metric in df_clustered.columns and metric in representative_row:
                metrics_values[metric] = representative_row[metric]

        cluster_metrics[cluster_id] = metrics_values

        signature = cluster_signatures[cluster_id]
        print(f"\nCluster {cluster_id} ({metric_source}):")
        print(f"  Representatives: {signature['representatives']}")
        tradeoff_str = ", ".join([f"{tp['benefit']} vs {tp['cost']}" for tp in signature['tradeoff_pairs']])
        print(f"  Trade-offs: {tradeoff_str if tradeoff_str else 'None'}")
    
    # Pareto dominance analysis
    efficient_clusters = set(range(n_clusters))
    dominated_by = {cid: set() for cid in range(n_clusters)}
    
    print("\n" + "-"*80)
    print("Pareto Dominance Analysis")
    print("-"*80)
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            
            sig_i = cluster_signatures[i]
            sig_j = cluster_signatures[j]
            metrics_i = cluster_metrics[i]
            metrics_j = cluster_metrics[j]
            
            # Check if cluster i dominates cluster j
            dominates = True
            
            # Check j's representatives
            for metric in sig_j['representatives']:
                if metric in metrics_i and metric in metrics_j:
                    direction = get_metric_directionality(df_clustered, metric)
                    if direction == 1:  # Higher is better
                        if metrics_i[metric] < metrics_j[metric]:
                            dominates = False
                            break
                    else:  # Lower is better
                        if metrics_i[metric] > metrics_j[metric]:
                            dominates = False
                            break
            
            # Check j's trade-offs (minimize cost metrics for dominance)
            if dominates and sig_j['tradeoff_pairs']:
                for tradeoff_pair in sig_j['tradeoff_pairs']:
                    cost_metric = tradeoff_pair['cost']
                    if cost_metric in metrics_i and cost_metric in metrics_j:
                        direction = get_metric_directionality(df_clustered, cost_metric)
                        # For costs, lower is better (minimize)
                        if direction == -1:  # Lower is better
                            if metrics_i[cost_metric] > metrics_j[cost_metric]:
                                dominates = False
                                break
                        else:
                            if metrics_i[cost_metric] < metrics_j[cost_metric]:
                                dominates = False
                                break
            
            if dominates:
                print(f"  Cluster {i} dominates Cluster {j}")
                if j in efficient_clusters:
                    efficient_clusters.discard(j)
                dominated_by[j].add(i)
    
    efficient_clusters = sorted(list(efficient_clusters))
    
    print(f"\n✓ Efficient Clusters (Pareto Frontier): {efficient_clusters}")
    print(f"✓ Dominated Clusters: {sorted([k for k, v in dominated_by.items() if v])}")
    
    # Analyze dominance relationships in detail - pass dominated_by to ensure consistency
    dominance_analysis = analyze_cluster_dominance(
        cluster_signatures, cluster_metrics, n_clusters, df_clustered, dominated_by
    )
    
    # Create frontier dataframe
    frontier_data = []
    for cluster_id in efficient_clusters:
        sig = cluster_signatures[cluster_id]
        metrics = cluster_metrics[cluster_id]
        
        frontier_data.append({
            'cluster_id': cluster_id,
            'efficiency_status': 'Efficient',
            'n_representatives': sig['n_representatives'],
            'n_tradeoff_pairs': sig['n_tradeoff_pairs'],
            'cluster_size': len(df_clustered[df_clustered['cluster'] == cluster_id])
        })
    
    for cluster_id in range(n_clusters):
        if cluster_id not in efficient_clusters:
            sig = cluster_signatures[cluster_id]
            dominators = list(dominated_by[cluster_id])
            
            frontier_data.append({
                'cluster_id': cluster_id,
                'efficiency_status': f'Dominated by {dominators}',
                'n_representatives': sig['n_representatives'],
                'n_tradeoff_pairs': sig['n_tradeoff_pairs'],
                'cluster_size': len(df_clustered[df_clustered['cluster'] == cluster_id])
            })
    
    pareto_frontier_df = pd.DataFrame(frontier_data)
    
    return {
        'efficient_clusters': efficient_clusters,
        'dominated_by': dominated_by,
        'cluster_signatures': cluster_signatures,
        'cluster_metrics': cluster_metrics,
        'pareto_frontier': pareto_frontier_df,
        'dominance_analysis': dominance_analysis
    }


def micro_ranking_hero_runs(df_clustered, insights, n_clusters, efficient_clusters, precomputed_hero_runs=None):
    """
    LEVEL 2: MICRO-RANKING (Local - Best Run per Cluster)
    
    For each cluster, identify the "Hero Run" that best maximizes Representatives
    while minimizing the impact of Trade-offs.
    
    Uses weighted distance scoring combining:
    - Distance to ideal on Representatives (α = 0.70)
    - Distance to ideal on Trade-offs (γ = 0.30)
    
    Returns:
        Dictionary with hero runs and rankings per cluster
    """
    print("\n" + "="*80)
    print("LEVEL 2: MICRO-RANKING (Best Run per Cluster - 'Hero Run')")
    print("="*80)
    
    hero_runs = {}
    cluster_rankings = {}
    
    for cluster_id in efficient_clusters:
        print(f"\nCluster {cluster_id} - Finding Hero Run...")
        
        signature = extract_cluster_signature(insights, cluster_id)
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            print(f"  ⚠ No data for cluster {cluster_id}")
            continue
        
        hero_run = None
        if precomputed_hero_runs is not None and cluster_id in precomputed_hero_runs:
            hero_run = precomputed_hero_runs.get(cluster_id)

        if hero_run is None:
            # Fallback to compute if not precomputed
            hero = compute_hero_run_for_cluster(signature, cluster_data, df_clustered)
            if hero:
                hero_run = {
                    'workflow_id': hero['workflow_id'],
                    'ensemble_score': hero['ensemble_score'],
                    'representatives_dist': hero['representatives_dist'],
                    'tradeoffs_dist': hero['tradeoffs_dist'],
                    'index': hero['row'].name if hasattr(hero['row'], 'name') else None
                }
        
        
        if hero_run:
            print(f"  Hero Run: {hero_run['workflow_id']}")
            print(f"    Representatives Distance: {hero_run['representatives_dist']:.4f}")
            print(f"    Trade-offs Distance: {hero_run['tradeoffs_dist']:.4f}")
            print(f"    Ensemble Score: {hero_run['ensemble_score']:.4f}")
            
            hero_runs[cluster_id] = hero_run
            # If rankings not available (because we reused), store minimal
            if precomputed_hero_runs is not None:
                cluster_rankings[cluster_id] = [hero_run]
            else:
                cluster_rankings[cluster_id] = []
        
    return {
        'hero_runs': hero_runs,
        'cluster_rankings': cluster_rankings
    }


def save_ranking_results(macro_results, micro_results, efficient_clusters, n_clusters, 
                        output_dir='data', filename_prefix='workflows', insights=None, borda_results=None, copeland_results=None,
                        export_csv=True, csv_dir=None, metric_source='mean'):
    """Save ranking results to CSV and JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    if csv_dir is None:
        csv_dir = os.path.join(output_dir, 'csv')
    if export_csv:
        os.makedirs(csv_dir, exist_ok=True)
    
    # 1. Save Pareto frontier
    if export_csv:
        pareto_file = os.path.join(csv_dir, f'{filename_prefix}_pareto_frontier.csv')
        macro_results['pareto_frontier'].to_csv(pareto_file, index=False)
        print(f"\n✓ Saved Pareto frontier to: {pareto_file}")
    
    # 1b. Save Borda ranking
    if export_csv and borda_results and 'borda_ranking' in borda_results and not borda_results['borda_ranking'].empty:
        borda_file = os.path.join(csv_dir, f'{filename_prefix}_borda_ranking.csv')
        borda_results['borda_ranking'].to_csv(borda_file, index=False)
        print(f"✓ Saved Borda ranking to: {borda_file}")
    
    # 1c. Save Copeland ranking
    if export_csv and copeland_results and 'copeland_ranking' in copeland_results and not copeland_results['copeland_ranking'].empty:
        copeland_file = os.path.join(csv_dir, f'{filename_prefix}_copeland_ranking.csv')
        copeland_results['copeland_ranking'].to_csv(copeland_file, index=False)
        print(f"✓ Saved Copeland ranking to: {copeland_file}")
    
    # 2. Save hero runs summary
    hero_data = []
    for cluster_id, hero_run in micro_results['hero_runs'].items():
        hero_data.append({
            'cluster_id': cluster_id,
            'hero_workflow_id': hero_run['workflow_id'],
            'ensemble_score': hero_run['ensemble_score'],
            'representatives_distance': hero_run['representatives_dist'],
            'tradeoffs_distance': hero_run['tradeoffs_dist']
        })
    
    if hero_data and export_csv:
        hero_df = pd.DataFrame(hero_data)
        hero_file = os.path.join(csv_dir, f'{filename_prefix}_hero_runs.csv')
        hero_df.to_csv(hero_file, index=False)
        print(f"✓ Saved hero runs to: {hero_file}")
    
    # 3. Save detailed per-cluster rankings
    if export_csv:
        for cluster_id, rankings in micro_results['cluster_rankings'].items():
            rankings_df = pd.DataFrame(rankings)
            rank_file = os.path.join(csv_dir, f'cluster_{cluster_id}_run_rankings.csv')
            rankings_df.to_csv(rank_file, index=False)
        print(f"✓ Saved per-cluster run rankings to: {csv_dir}/cluster_X_run_rankings.csv")
    
    # 4. Save comprehensive ranking JSON with cluster signatures and dominance analysis
    cluster_signatures = {}
    dominance_analysis = macro_results.get('dominance_analysis', {})
    
    for cluster_id in range(n_clusters):
        sig = extract_cluster_signature(insights, cluster_id)
        dom_report = dominance_analysis.get(cluster_id, {})
        
        # Serialize strengths and weaknesses
        strengths_serialized = []
        for s in dom_report.get('strengths', []):
            strengths_serialized.append({
                'metric': s['metric'],
                'value': float(s['value']),
                'direction': s['direction'],
                'value_category': s.get('value_category', 'high'),
                'beats_n_clusters': s['beats_n_clusters'],
                'beats_by_default': s.get('beats_by_default', 0),
                'beats_in_competition': s.get('beats_in_competition', 0),
                'avg_improvement_pct': float(s['avg_improvement_pct'])
            })
        
        weaknesses_serialized = []
        for w in dom_report.get('weaknesses', []):
            weaknesses_serialized.append({
                'metric': w['metric'],
                'value': float(w['value']),
                'direction': w['direction'],
                'value_category': w.get('value_category', 'high'),
                'worse_than_n_clusters': w.get('worse_than_n_clusters', 0),
                'reason': w.get('reason', 'performance')
            })
        
        # Serialize dominance relationships
        dominates_serialized = []
        for dom in dom_report.get('dominates_others', []):
            reasons_serialized = []
            for r in dom['reasons']:
                reason_dict = {
                    'metric': r['metric'],
                    'type': r['type'],
                    'cluster_value': float(r['cluster_i_value']),
                    'dominated_cluster_value': float(r['cluster_j_value']),
                    'improvement_pct': float(r['improvement_pct']),
                    'direction': r['direction']
                }
                if r['type'] == 'tradeoff_cost':
                    reason_dict['related_benefit'] = r['related_benefit']
                reasons_serialized.append(reason_dict)
            
            dominates_serialized.append({
                'dominated_cluster': dom['dominated_cluster'],
                'reasons': reasons_serialized
            })
        
        dominated_by_serialized = []
        for dom in dom_report.get('dominated_by', []):
            dominated_by_serialized.append({
                'dominating_cluster': dom['dominating_cluster'],
                'n_reasons': len(dom['reasons'])
            })
        
        cluster_signatures[str(int(cluster_id))] = {
            'representatives': sig['representatives'],
            'representative_value_categories': sig.get('representative_value_categories', {}),
            'tradeoff_pairs': [
                {
                    'benefit': tp['benefit'],
                    'cost': tp['cost'],
                    'correlation': float(tp['correlation'])
                }
                for tp in sig['tradeoff_pairs']
            ],
            'n_representatives': sig['n_representatives'],
            'n_tradeoff_pairs': sig['n_tradeoff_pairs'],
            'is_efficient': cluster_id in efficient_clusters,
            'strengths': strengths_serialized,
            'weaknesses': weaknesses_serialized,
            'dominates': dominates_serialized,
            'dominated_by': dominated_by_serialized
        }
    
    ranking_summary = {
        'macro_ranking': {
            'efficient_clusters': [int(cid) for cid in efficient_clusters],
            'n_efficient': int(len(efficient_clusters)),
            'n_total_clusters': int(n_clusters),
            'efficiency_ratio': float(len(efficient_clusters) / n_clusters),
            'metric_source': metric_source
        },
        'consensus_rankings': {
            'borda': {
                'ranking': borda_results['borda_ranking'].to_dict('records') if borda_results else [],
                'methodology': 'Borda count: best cluster on each metric gets (n_clusters-1) points, worst gets 0'
            },
            'copeland': {
                'ranking': copeland_results['copeland_ranking'].to_dict('records') if copeland_results else [],
                'methodology': 'Copeland: head-to-head pairwise comparisons, winner of each pair gets 1 point'
            }
        },
        'cluster_signatures': cluster_signatures,
        'micro_ranking': {
            'hero_runs': {
                str(int(cid)): {
                    'workflow_id': str(run['workflow_id']),
                    'ensemble_score': float(run['ensemble_score']),
                    'representatives_distance': float(run['representatives_dist']),
                    'tradeoffs_distance': float(run['tradeoffs_dist'])
                }
                for cid, run in micro_results['hero_runs'].items()
            }
        }
    }
    
    json_file = os.path.join(output_dir, f'{filename_prefix}_ranking_summary.json')
    with open(json_file, 'w') as f:
        json.dump(ranking_summary, f, indent=2)
    
    print(f"✓ Saved ranking summary JSON to: {json_file}")


def main():
    """Main execution function."""
    print("="*80)
    print("CLUSTER RANKING & SELECTION PIPELINE")
    print("Level 1: Macro-Ranking (Global Efficiency via Pareto Dominance)")
    print("Level 2: Micro-Ranking (Best Run per Cluster - 'Hero Run')")
    print("="*80)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python 3_cluster_ranking.py <data_folder> [--no-csv] [--metric-sources=mean,medoid,hero|all]")
        print("\nArguments:")
        print("  data_folder           - Path to folder containing clustering results and insights")
        print("  --no-csv              - Skip writing CSV outputs (JSON still generated)")
        print("  --metric-sources=...  - Comma-separated sources: mean (default), medoid, hero, or 'all'")
        sys.exit(1)

    data_folder = sys.argv[1]
    export_csv = '--no-csv' not in sys.argv

    metric_sources_arg = next((arg for arg in sys.argv if arg.startswith('--metric-sources=')), None)
    if metric_sources_arg:
        metric_sources = [src.strip().lower() for src in metric_sources_arg.split('=', 1)[1].split(',') if src.strip()]
        if 'all' in metric_sources:
            metric_sources = ['mean', 'medoid', 'hero']
    else:
        metric_sources = ['hero']

    metric_sources = list(dict.fromkeys(metric_sources))  # de-duplicate, preserve order
    multiple_sources = len(metric_sources) > 1
    
    # Load data
    print("\nLoading cluster data and insights...")
    df_clustered, insights, df_model_eval, df_medoids = load_cluster_data(data_folder)
    print(f"✓ Loaded {len(df_clustered)} workflows in {df_clustered['cluster'].max() + 1} clusters")
    
    n_clusters = df_clustered['cluster'].max() + 1
    for metric_source in metric_sources:
        print("\n" + "#"*80)
        print(f"USING METRIC SOURCE: {metric_source.upper()}")
        print("#"*80)

        csv_dir = os.path.join(data_folder, 'csv')
        if export_csv:
            os.makedirs(csv_dir, exist_ok=True)

        # Precompute hero runs once (for reuse and for hero metric source)
        precomputed_hero_runs = {}
        for cid in range(n_clusters):
            sig = extract_cluster_signature(insights, cid)
            cluster_data = df_clustered[df_clustered['cluster'] == cid]
            hero = compute_hero_run_for_cluster(sig, cluster_data, df_clustered)
            if hero:
                precomputed_hero_runs[cid] = {
                    'workflow_id': hero['workflow_id'],
                    'ensemble_score': hero['ensemble_score'],
                    'representatives_dist': hero['representatives_dist'],
                    'tradeoffs_dist': hero['tradeoffs_dist'],
                    'index': hero['row'].name if hasattr(hero['row'], 'name') else None
                }

        # LEVEL 1: MACRO-RANKING
        macro_results = macro_ranking_pareto_dominance(df_clustered, insights, n_clusters, df_medoids, metric_source)
        efficient_clusters = macro_results['efficient_clusters']

        # BORDA COUNT RANKING (Consensus ranking across all representative metrics)
        borda_results = borda_ranking_clusters(
            macro_results['cluster_signatures'],
            macro_results['cluster_metrics'],
            n_clusters,
            df_clustered
        )

        # COPELAND'S RANKING (Head-to-head pairwise comparisons)
        copeland_results = copeland_ranking_clusters(
            macro_results['cluster_signatures'],
            macro_results['cluster_metrics'],
            n_clusters,
            df_clustered
        )

        # LEVEL 2: MICRO-RANKING (only for efficient clusters)
        micro_results = micro_ranking_hero_runs(df_clustered, insights, n_clusters, efficient_clusters, precomputed_hero_runs)

        # Determine file prefix (avoid collisions when multiple sources requested)
        base_prefix = os.path.basename(data_folder)
        filename_prefix = f"{base_prefix}_{metric_source}" if multiple_sources else base_prefix

        # Save results (include borda_results and copeland_results)
        save_ranking_results(
            macro_results,
            micro_results,
            efficient_clusters,
            n_clusters,
            data_folder,
            filename_prefix,
            insights,
            borda_results,
            copeland_results,
            export_csv,
            csv_dir,
            metric_source
        )

        # Print summary
        print("\n" + "="*80)
        print(f"RANKING PIPELINE SUMMARY (metric source: {metric_source})")
        print("="*80)
        print(f"\nGlobal Analysis (Macro-Ranking):")
        print(f"  Total Clusters: {n_clusters}")
        print(f"  Efficient Clusters: {len(efficient_clusters)}")
        print(f"  Efficiency Ratio: {len(efficient_clusters) / n_clusters:.1%}")

        print(f"\nBorda Consensus Ranking (Top 5):")
        if 'borda_ranking' in borda_results and not borda_results['borda_ranking'].empty:
            top_5 = borda_results['borda_ranking'].head(5)
            for _, row in top_5.iterrows():
                print(f"  #{int(row['borda_rank'])}: Cluster {int(row['cluster_id'])} (score: {int(row['borda_score'])}, size: {int(row['cluster_size'])})")

        print(f"\nCopeland's Ranking (Top 5):")
        if 'copeland_ranking' in copeland_results and not copeland_results['copeland_ranking'].empty:
            top_5 = copeland_results['copeland_ranking'].head(5)
            for _, row in top_5.iterrows():
                print(f"  #{int(row['copeland_rank'])}: Cluster {int(row['cluster_id'])} (score: {row['copeland_score']:.1f}, size: {int(row['cluster_size'])})")

        print(f"\nBest Clusters (Pareto Frontier):")
        for cid in efficient_clusters:
            print(f"  - Cluster {cid} (size: {len(df_clustered[df_clustered['cluster'] == cid])})")

        print(f"\nBest Runs (Micro-Ranking - Hero Runs):")
        for cluster_id, hero_run in micro_results['hero_runs'].items():
            print(f"  Cluster {cluster_id}: {hero_run['workflow_id']} (score: {hero_run['ensemble_score']:.4f})")

        print("\n" + "="*80)
        if export_csv:
            print("Output files generated in folder:")
            print(f"  {csv_dir}/")
            print(f"  - {filename_prefix}_pareto_frontier.csv (macro-ranking results)")
            print(f"  - {filename_prefix}_borda_ranking.csv (Borda consensus ranking)")
            print(f"  - {filename_prefix}_copeland_ranking.csv (Copeland's pairwise ranking)")
            print(f"  - {filename_prefix}_hero_runs.csv (best runs per cluster)")
            print("  - cluster_X_run_rankings.csv (per-cluster run rankings)")
        else:
            print("CSV export disabled (--no-csv). Skipped writing CSV outputs.")
        print(f"  - {filename_prefix}_ranking_summary.json (comprehensive ranking summary)")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
