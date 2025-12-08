"""
ML Workflow Cluster Classification & Feature Importance Analysis
=================================================================
Implementation of classification methodology following paper guidelines:

PHASE 1: CLUSTER-SPECIFIC CLASSIFICATION ANALYSIS
  Step 1: Load clustering results and prepare data
  Step 2: Feature selection on original data (correlation, SHAP-based)
  Step 3: Train & evaluate XGBoost models using selected features
  Step 4: Local and global SHAP interpretability analysis

Pipeline structure allows enabling/disabling steps for testing.
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import f_oneway
import shap
from _correlation_utils import (
    detect_variable_types,
    compute_relationship_measure,
    compute_relationship_matrix
)
import warnings
import json
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class InsightsPipeline:
    """
    Modular pipeline for ML workflow classification & feature selection.
    Allows enabling/disabling and reordering steps for testing.
    Steps can be optional and dependent steps handle missing results gracefully.
    """
    
    def __init__(self):
        """Initialize pipeline state."""
        self.steps = []
        self.results = {}
        self.enabled_steps = set()
        self.skip_missing_deps = True
    
    def add_step(self, name, func, enabled=True):
        """Add a step to the pipeline."""
        self.steps.append({'name': name, 'func': func})
        if enabled:
            self.enabled_steps.add(name)
        print(f"✓ Added step: {name} (enabled={enabled})")
    
    def enable_step(self, name):
        """Enable a specific step."""
        if name in [s['name'] for s in self.steps]:
            self.enabled_steps.add(name)
            print(f"✓ Enabled: {name}")
        else:
            print(f"✗ Step not found: {name}")
    
    def disable_step(self, name):
        """Disable a specific step."""
        self.enabled_steps.discard(name)
        print(f"✓ Disabled: {name}")
    
    def set_steps(self, step_names):
        """Enable only specific steps (disable all others)."""
        self.enabled_steps = set(step_names)
        print(f"✓ Pipeline set to run only: {', '.join(step_names)}")
    
    def list_steps(self):
        """Print all available steps and their status."""
        print("\n" + "="*60)
        print("Pipeline Steps")
        print("="*60)
        for step in self.steps:
            status = "✓ ENABLED" if step['name'] in self.enabled_steps else "✗ DISABLED"
            print(f"  {status:12} - {step['name']}")
    
    def get_result(self, step_name, key=None, default=None):
        """Safely retrieve results from a previous step."""
        if step_name not in self.results:
            return default
        
        result = self.results[step_name]
        if key is None:
            return result
        
        if isinstance(result, dict):
            return result.get(key, default)
        
        return default
    
    def run(self, **kwargs):
        """Execute the pipeline, running only enabled steps in order."""
        print("\n" + "="*60)
        print("ML Insights & Classification Pipeline")
        print("="*60)
        
        for step in self.steps:
            step_name = step['name']
            
            if step_name not in self.enabled_steps:
                print(f"\n⊘ Skipping: {step_name}")
                continue
            
            print(f"\n▶ Running: {step_name}")
            print("-" * 60)
            
            try:
                result = step['func'](self.results, self, **kwargs)
                self.results[step_name] = result
                print(f"✓ Completed: {step_name}")
            except KeyError as e:
                print(f"✗ Error in {step_name}: Missing dependency {e}")
                if self.skip_missing_deps:
                    print(f"  Skipping this step due to missing results from a disabled step.")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"✗ Error in {step_name}: {e}")
                raise
        
        return self.results


def load_clustering_results(data_folder='data', prefix='workflows'):
    """Load clustering output files from data folder."""
    df_clustered = pd.read_csv(os.path.join(data_folder, f'{prefix}_clustered.csv'))
    medoids = pd.read_csv(os.path.join(data_folder, f'{prefix}_medoids.csv'))
    
    # Load cluster metadata to identify small/outlier clusters
    try:
        cluster_metadata = pd.read_csv(os.path.join(data_folder, f'{prefix}_cluster_metadata.csv'))
    except FileNotFoundError:
        cluster_metadata = None

    print(f"Loaded clustering results from '{data_folder}':")
    print(f"  - {len(df_clustered)} workflows in {df_clustered['cluster'].nunique()} clusters")
    
    if cluster_metadata is not None:
        small_clusters = cluster_metadata[cluster_metadata['is_small'] == True]['cluster_id'].tolist()
        if small_clusters:
            print(f"  - Small/outlier clusters detected: {small_clusters}")

    return df_clustered, medoids, cluster_metadata


def load_processed_data(data_folder='data', prefix='workflows'):
    """Load pre-processed (dimensionality reduced) data for Phase 2."""
    try:
        X_processed_df = pd.read_csv(os.path.join(data_folder, f'{prefix}_processed_data.csv'))
        print(f"Loaded processed data: {X_processed_df.shape}")
        return X_processed_df
    except FileNotFoundError:
        print(f"⚠ Processed data file not found: {os.path.join(data_folder, f'{prefix}_processed_data.csv')}")
        return None


def feature_selection_shap_iterative(X_train, y_train, feature_names, n_iterations, correlation_threshold):
    """
    Steps 1-3: Iterative Feature Selection with SHAP + Correlation Removal.
    
    Following paper methodology:
    "First, if some features are to be kept because of their relevance to the study, all other 
    features with a strong relationship to the former are removed. Secondly, a random forest 
    classifier is iteratively trained. At each iteration, the feature with the highest SHAP 
    value not previously visited is selected and all other highly related features are removed. 
    
    Returns:
    - final_selected: List of final selected features
    - removal_analysis: DataFrame with feature removal metrics
    - selection_history: List of selected features with their removed correlates
    
    Process:
    1. Train RF on all remaining features
    2. Get SHAP values
    3. Select highest SHAP feature (not previously selected)
    4. Find ALL features related to it (using multi-type correlation)
    5. Remove those related features from remaining set
    6. REPEAT until convergence or n_iterations reached
    """
    print("\n" + "="*80)
    print("STEPS 1-3: Iterative SHAP Selection + Correlation Removal")
    print("="*80)
    print("Following paper: Iterative RF + SHAP ranking + correlation-based removal per iteration")
    print(f"Correlation threshold for removal: {correlation_threshold}")
    
    if n_iterations is None:
        n_iterations = min(10, len(feature_names) // 2) 
    
    selected_features = []
    previously_selected = set()
    remaining_features = list(feature_names)
    iteration = 0
    
    # Track selection history for feature removal analysis
    selection_history = []
    
    # Track removed features with their correlations for later analysis
    removed_features_analysis = {}  # {removed_feature: {'max_relationship': float, 'related_to': selected_feature, 'all_relationships': [...]}}
    
    # ============================================================================
    # Iterative SHAP-based selection with correlation removal
    # ============================================================================
    print("\n" + "-"*80)
    print("Iterative SHAP-based Feature Selection + Correlation Removal")
    print("-"*80)
    
    while len(selected_features) < n_iterations and len(remaining_features) > 1:
        iteration += 1
        
        print(f"\n  Iteration {iteration}:")
        print(f"    Training RF on {len(remaining_features)} remaining features...")
        
        # Step 1: Train random forest on REMAINING features only
        X_temp = X_train[:, [feature_names.index(f) for f in remaining_features]]
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_temp, y_train)
        
        # Step 2: Get SHAP importance for current feature set
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_temp)
        except Exception as e:
            print(f"    Warning: SHAP failed, using tree importance fallback")
            shap_importance = rf.feature_importances_
            importance_with_idx = [(i, shap_importance[i], remaining_features[i]) 
                                   for i in range(len(remaining_features))]
        else:
            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            
            if len(shap_vals.shape) == 3:
                shap_importance = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                shap_importance = np.abs(shap_vals).mean(axis=0)
            
            # Create list of (index, importance, feature_name) tuples
            importance_with_idx = [(i, shap_importance[i], remaining_features[i]) 
                                   for i in range(len(remaining_features))]
        
        # Sort by importance, descending
        importance_with_idx.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Find highest SHAP feature not previously selected
        most_important_feature = None
        most_important_shap = 0
        
        for idx, shap_val, feat in importance_with_idx:
            if feat not in previously_selected:
                most_important_feature = feat
                most_important_shap = shap_val
                break
        
        # If all remaining features were already selected, stop
        if most_important_feature is None:
            print(f"    All remaining features already selected. Stopping.")
            break
        
        selected_features.append(most_important_feature)
        previously_selected.add(most_important_feature)
        
        print(f"    Selected: '{most_important_feature}' (SHAP: {most_important_shap:.6f})")
        
        # Step 4 & 5: Find and remove highly related features using multi-type correlation
        print(f"    Removing features related to '{most_important_feature}'...")
        
        # Compute relationship matrix for remaining features
        X_remaining = X_train[:, [feature_names.index(f) for f in remaining_features]]
        relationship_matrix = compute_relationship_matrix(X_remaining, remaining_features)
        
        # Get relationships to the selected feature
        related_to_selected = relationship_matrix[most_important_feature]
        # Find features exceeding correlation threshold (excluding the selected feature itself)
        to_remove = [f for f in remaining_features 
                     if f != most_important_feature 
                     and related_to_selected.get(f, 0) > correlation_threshold]
        to_remove.append(most_important_feature)  # Also remove the selected feature from remaining

        # Track removed features with their relationships
        removed_with_relations = [(f, related_to_selected.get(f, 0)) for f in to_remove]
        removed_with_relations.sort(key=lambda x: x[1], reverse=True)
        
        # Build removed features analysis (excluding the selected feature itself)
        removed_features_only = [f for f in to_remove if f != most_important_feature]
        for removed_feat, relation_measure in removed_with_relations:
            if removed_feat != most_important_feature:  # Skip the selected feature
                if removed_feat not in removed_features_analysis:
                    removed_features_analysis[removed_feat] = {
                        'max_relationship': relation_measure,
                        'related_to': most_important_feature,
                        'all_relationships': []
                    }
                else:
                    # Update max_relationship if this one is stronger
                    if relation_measure > removed_features_analysis[removed_feat]['max_relationship']:
                        removed_features_analysis[removed_feat]['max_relationship'] = relation_measure
                        removed_features_analysis[removed_feat]['related_to'] = most_important_feature
                
                # Track all relationships for this removed feature
                removed_features_analysis[removed_feat]['all_relationships'].append({
                    'selected_feature': most_important_feature,
                    'relationship_measure': relation_measure
                })
        
        selection_history.append({
            'iteration': iteration,
            'selected_feature': most_important_feature,
            'shap_importance': most_important_shap,
            'features_removed': to_remove,
            'removed_count': len(to_remove),
            'removed_details': removed_with_relations  # Keep for detailed tracking
        })
        
        if to_remove:
            print(f"    Removing {len(to_remove)} related features (measure > {correlation_threshold}):")
            for removed_feat, relation in removed_with_relations[:5]:  # Show first 5
                print(f"      - {removed_feat} (relation: {relation:.4f})")
            if len(to_remove) > 5:
                print(f"      ... and {len(to_remove) - 5} more")
        
        # Step 6: Update remaining features for next iteration
        remaining_features = [f for f in remaining_features if f not in to_remove]
    
    print(f"\n✓ After iterative selection: {len(selected_features)} features selected")
    
    return selected_features, selection_history, removed_features_analysis


def main():
    """Main execution function """
    print("="*80)
    print("ML WORKFLOW CLASSIFICATION & INTERPRETABILITY")
    print("  PHASE 1: Feature Selection on Original Data (all features)")
    print("  PHASE 2: Model Training on Selected Features (future: with PCA-clustered data)")
    print("="*80 + "\n")

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python 2_generate_cluster_insights.py <data_folder> [--no-csv]")
        print("\nArguments:")
        print("  data_folder - Path to folder containing clustering results and parameter/metric files")
        print("  --no-csv    - Skip writing CSV outputs (JSON still generated)")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    export_csv = '--no-csv' not in sys.argv
    csv_dir = os.path.join(data_folder, 'csv')
    if export_csv:
        os.makedirs(csv_dir, exist_ok=True)

    # Load clustering results and processed data
    df_clustered, medoids, cluster_metadata = load_clustering_results(data_folder)
    X_processed_df = load_processed_data(data_folder)

    # Load metric and parameter names from files (required - no fallback)
    metrics_file = os.path.join(data_folder, 'metric_names.txt')
    params_file = os.path.join(data_folder, 'parameter_names.txt')
    
    # Read metric columns from file
    with open(metrics_file, 'r') as f:
        metric_cols = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✓ Loaded metric names from {metrics_file}")
    
    # Read parameter names from file
    with open(params_file, 'r') as f:
        param_cols = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✓ Loaded parameter names from {params_file}")
    
    # Filter metric_cols to only include columns that exist in df_clustered
    available_cols = set(df_clustered.columns)
    missing_metric_cols = [col for col in metric_cols if col not in available_cols]
    metric_cols = [col for col in metric_cols if col in available_cols]
    
    if missing_metric_cols:
        print(f"\n⚠ Warning: {len(missing_metric_cols)} metric columns not found in clustered data:")
        for col in missing_metric_cols:
            print(f"    - {col}")
        print(f"✓ Filtered to {len(metric_cols)} available metric columns")
    
    if not metric_cols:
        print("✗ Error: No metric columns found in dataframe! Cannot proceed.")
        sys.exit(1)
    
    print(f"\nTotal features to analyze: {len(metric_cols)}")
    print(f"Features: {metric_cols}\n")

    # Extract small clusters from metadata
    small_clusters = set()
    if cluster_metadata is not None:
        small_clusters = set(cluster_metadata[cluster_metadata['is_small'] == True]['cluster_id'].tolist())

    # Prepare original data - extract metric columns and standardize
    print("Preparing data: Standardizing original metrics...")
    X_original = df_clustered[metric_cols].copy()
    
    # Standardize using StandardScaler (same as clustering pipeline)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_original)
    
    print(f"✓ Data standardized: shape {X_standardized.shape}")
    
    cluster_labels = df_clustered['cluster'].values
    n_clusters = df_clustered['cluster'].max() + 1

    # Build the pipeline
    pipeline = build_default_insights_pipeline()
    
    # Print current pipeline configuration
    pipeline.list_steps()
    
    # Configure pipeline parameters
    pipeline_params = {
        'df_clustered': df_clustered,
        'medoids': medoids,
        'X_standardized': X_standardized,
        'X_processed_df': X_processed_df,
        'metric_cols': metric_cols,
        'param_cols': param_cols,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'small_clusters': small_clusters,
        'correlation_threshold': 0.75,
        'n_iterations': None,
        'data_folder': data_folder,
        'csv_dir': csv_dir,
        'export_csv': export_csv
    }
    
    # Run the pipeline
    results = pipeline.run(**pipeline_params)

    # Print summary table
    print("\n" + "="*80)
    print("PHASE 1 RESULTS SUMMARY")
    print("="*80)
    results_summary = pipeline.get_result('step_phase1_feature_selection', key='results_summary')
    if results_summary is not None and not results_summary.empty:
        print(results_summary[['cluster_id', 'n_samples', 'n_features_selected']].to_string(index=False))
    
    # Print model evaluation summary
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    model_summary = pipeline.get_result('step_phase1_model_training', key='models_summary')
    if model_summary is not None and not model_summary.empty:
        # Include balanced accuracy for predictive quality
        cols = [c for c in ['cluster_id', 'n_samples', 'balanced_accuracy', 'f1_score', 'test_auc'] if c in model_summary.columns]
        print(model_summary[cols].to_string(index=False))

    print("\n" + "="*80)
    if export_csv:
        print("Analysis Complete! Generated Files in folder:")
        print(f"  {csv_dir}/")
        print("="*80)
        print("  PHASE 1 - Feature Selection:")
        print("    - workflows_classification_results.csv (per-cluster summary)")
        print("    - cluster_X_selection_history.csv (feature removal tracking)")
        print("    - cluster_X_tradeoff_analysis.csv (negative correlations/trade-offs)")
        print("  PHASE 1 - Model Training:")
        print("    - workflows_model_evaluation_summary.csv (per-cluster model performance)")
        print("  PHASE 1 - Decision Tree Rules:")
        print("    - cluster_decision_rules.csv (interpretable decision paths per workflow)")
    else:
        print("CSV export disabled (--no-csv). Skipped writing CSV outputs.")
    print("  PHASE 1 - Comprehensive Statistics:")
    print("    - clusters_comprehensive_insights.json (all important insights per cluster)")
    print("="*80 + "\n")


# ============ PIPELINE STEPS - PHASE 1: FEATURE SELECTION ============

def step_phase1_load_data(results, pipeline, **kwargs):
    """Step 1.1: Load clustering results and prepare data."""
    df_clustered = kwargs.get('df_clustered')
    medoids = kwargs.get('medoids')
    X_standardized = kwargs.get('X_standardized')
    X_processed_df = kwargs.get('X_processed_df')
    metric_cols = kwargs.get('metric_cols')
    param_cols = kwargs.get('param_cols')
    cluster_labels = kwargs.get('cluster_labels')
    n_clusters = kwargs.get('n_clusters')
    small_clusters = kwargs.get('small_clusters', set())
    data_folder = kwargs.get('data_folder', 'data')
    csv_dir = kwargs.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = kwargs.get('export_csv', True)
    
    return {
        'df_clustered': df_clustered,
        'medoids': medoids,
        'X_standardized': X_standardized,
        'X_processed_df': X_processed_df,
        'metric_cols': metric_cols,
        'param_cols': param_cols,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'small_clusters': small_clusters,
        'data_folder': data_folder,
        'csv_dir': csv_dir,
        'export_csv': export_csv
    }


def step_phase1_feature_selection(results, pipeline, **kwargs):
    """
    Step 1.2-1.3: PHASE 1 - Feature Selection on Original Data
    
    - Multi-step feature selection (correlation, SHAP-based)
    - Track removed features
    """
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    df_clustered = load_result.get('df_clustered')
    X_standardized = load_result.get('X_standardized')
    metric_cols = load_result.get('metric_cols')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    data_folder = load_result.get('data_folder', 'data')
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)

    correlation_threshold = kwargs.get('correlation_threshold', 0.75)
    n_iterations = kwargs.get('n_iterations', None)
    
    if any(v is None for v in [X_standardized, metric_cols, cluster_labels]):
        raise KeyError("step_phase1_load_data: Required data not loaded")
    
    print("\n" + "="*80)
    print("PHASE 1: FEATURE SELECTION ON ORIGINAL DATA (All Features - Standardized)")
    print("="*80)
    
    all_results = []
    correlation_analysis_per_cluster = {}  # Store correlation analysis for removed features
    removed_features_analysis_per_cluster = {}  # Store detailed removed features analysis
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        n_cluster = np.sum(cluster_mask)
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id} (n={n_cluster} workflows)", end="")
        
        if cluster_id in small_clusters:
            print(f" [SMALL - SKIPPED]")
            continue
        
        print()
        print(f"{'='*80}")
        
        # Create binary classification: this cluster vs. others
        y_binary = (cluster_labels == cluster_id).astype(int)
        
    
        X_train = X_standardized
        y_train = y_binary
        
        # print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
        # print(f"Class distribution (train): {np.bincount(y_train)}")
        # print(f"Class distribution (test): {np.bincount(y_test)}")
        
        # Feature selection (Steps 1-3 from paper)
        selected_features, selection_history, removed_features_analysis = feature_selection_shap_iterative(
            X_train, y_train, metric_cols, n_iterations, correlation_threshold
        )
        
        print(f"\n✓ Final selected features ({len(selected_features)}): {selected_features}")
        
        # Store removed features analysis for later use in comprehensive insights
        removed_features_analysis_per_cluster[cluster_id] = removed_features_analysis
        
        # Store results
        cluster_result = {
            'cluster_id': cluster_id,
            'n_samples': n_cluster,
            'n_features_selected': len(selected_features),
            'selected_features': ','.join(selected_features),
        }
        all_results.append(cluster_result)
        
        # Save selection history with correlation data already embedded
        selection_history_expanded = []
        for entry in selection_history:
            for removed_feat, relation_measure in entry['removed_details']:
                selection_history_expanded.append({
                    'iteration': entry['iteration'],
                    'selected_feature': entry['selected_feature'],
                    'selected_shap_importance': entry['shap_importance'],
                    'removed_feature': removed_feat,
                    'relationship_measure': relation_measure
                })
        
        if selection_history_expanded:
            selection_df = pd.DataFrame(selection_history_expanded)
            if export_csv:
                selection_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_selection_history.csv'), index=False)
                print(f"✓ Saved cluster_{cluster_id}_selection_history.csv ({len(selection_df)} removed features tracked)")
        
        # Extract correlation analysis from selection history (removed features were correlated to selected by design)
        print("\n" + "-"*80)
        print(f"Extracting Correlation Analysis for Removed Features")
        print("-"*80)
        
        correlation_results = []
        for entry in selection_history:
            selected_feat = entry['selected_feature']
            for removed_feat, relation_measure in entry['removed_details']:
                if removed_feat != selected_feat:  # Skip the selected feature itself
                    correlation_results.append({
                        'removed_feature': removed_feat,
                        'selected_feature': selected_feat,
                        'relationship_strength': relation_measure
                    })
        
        if correlation_results:
            # Store correlation analysis for JSON output
            correlation_analysis_per_cluster[cluster_id] = correlation_results
            
            print(f"\n✓ Extracted {len(correlation_results)} removed-to-selected feature correlations from feature selection")
            print("\nTop Correlations (Removed vs Selected):")
            sorted_corr = sorted(correlation_results, key=lambda x: x['relationship_strength'], reverse=True)
            for idx, corr in enumerate(sorted_corr[:10], 1):
                print(f"  {corr['removed_feature']} (removed) ↔ {corr['selected_feature']} (selected)")
                print(f"    Relationship Strength: {corr['relationship_strength']:.4f}\n")
        else:
            print(f"\n⚠ No correlation analysis available")
            correlation_analysis_per_cluster[cluster_id] = []
    
    # Create summary results dataframe
    if all_results:
        results_summary = pd.DataFrame(all_results)
        if export_csv:
            results_file = os.path.join(csv_dir, 'workflows_classification_results.csv')
            results_summary.to_csv(results_file, index=False)
            print(f"\n✓ Saved classification results summary to: {results_file}")
    else:
        results_summary = pd.DataFrame()
    
    return {
        'results_summary': results_summary,
        'selected_features_per_cluster': {r['cluster_id']: r['selected_features'] for r in all_results},
        'correlation_analysis_per_cluster': correlation_analysis_per_cluster,
        'removed_features_analysis_per_cluster': removed_features_analysis_per_cluster
    }


# ============ PIPELINE STEPS - PHASE 1 CONTINUED: TRADE-OFF ANALYSIS ============

def step_phase1_tradeoff_analysis(results, pipeline, **kwargs):
    """
    Step 1.3: PHASE 1 - Trade-off Analysis (Selected vs Non-Selected Features)
    
    For each cluster:
    - Identify selected vs non-selected features
    - Compute relationship measures between feature pairs
    - Save trade-off analysis: selected features vs non-selected features
    - Track negative correlations as trade-offs
    """
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    
    X_standardized = load_result.get('X_standardized')
    metric_cols = load_result.get('metric_cols')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    data_folder = load_result.get('data_folder', 'data')
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)

    selected_features_dict = feature_result.get('selected_features_per_cluster', {})
    correlation_threshold = kwargs.get('correlation_threshold', 0.75)
    
    if any(v is None for v in [X_standardized, metric_cols, cluster_labels]):
        raise KeyError("step_phase1_load_data: Required data not loaded for trade-off analysis")
    
    print("\n" + "="*80)
    print("PHASE 1: TRADE-OFF ANALYSIS (Selected vs Non-Selected Features)")
    print("="*80)
    
    tradeoff_analysis_per_cluster = {}
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            print(f"\n⊘ Cluster {cluster_id}: SKIPPED (small cluster)")
            tradeoff_analysis_per_cluster[cluster_id] = []
            continue
        
        cluster_mask = cluster_labels == cluster_id
        
        # Get selected features for this cluster
        selected_feat_str = selected_features_dict.get(cluster_id, '')
        if not selected_feat_str:
            print(f"\n⊘ Cluster {cluster_id}: No features selected, skipping trade-off analysis")
            tradeoff_analysis_per_cluster[cluster_id] = []
            continue
        
        selected_features = [f.strip() for f in selected_feat_str.split(',') if f.strip()]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}: Trade-off Analysis")
        print(f"{'='*80}")
        
        # Extract cluster data
        X_cluster = X_standardized[cluster_mask]
        X_cluster_df = pd.DataFrame(X_cluster, columns=metric_cols)
        
        var_types = detect_variable_types(X_cluster_df)
        correlation_results = []
        
        # Separate selected and non-selected features
        non_selected_features = [f for f in metric_cols if f not in selected_features]
        
        print(f"Selected features: {len(selected_features)}")
        print(f"Non-selected features: {len(non_selected_features)}")
        print(f"Analyzing pairs: selected vs non-selected (excluding non-selected vs non-selected)")
        
        # Analyze pairs: selected vs non-selected
        for feat1 in selected_features:
            for feat2 in non_selected_features:
                try:
                    measure, measure_type = compute_relationship_measure(
                        X_cluster_df, feat1, feat2, var_types
                    )
                    
                    x = X_cluster_df[feat1].values
                    y = X_cluster_df[feat2].values
                    print(f"  {feat1} ↔ {feat2} | Measure: {measure:.4f} ({measure_type})")
                    # Only include negative relationships as trade-offs
                    if measure < 0:
                        correlation_results.append({
                            'metric_1': feat1,
                            'metric_2': feat2,
                            'relationship_type': measure_type,
                            'relationship_strength': measure,
                            'is_tradeoff': 1 if measure < - correlation_threshold else 0
                        })
                except Exception as e:
                    pass
        
        if correlation_results:
            tradeoff_df = pd.DataFrame(correlation_results)
            tradeoff_df = tradeoff_df.sort_values('relationship_strength', ascending=False)
            if export_csv:
                tradeoff_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_tradeoff_analysis.csv'), index=False)
            
            # Store trade-off data for JSON output
            tradeoff_analysis_per_cluster[cluster_id] = tradeoff_df.to_dict('records')
            
            print(f"\n✓ Found {len(tradeoff_df)} negative correlations (trade-offs) in Cluster {cluster_id}")
            if export_csv:
                print(f"✓ Saved cluster_{cluster_id}_tradeoff_analysis.csv")
            
            print("\nTop Trade-off Relationships (Strongest Negative Correlations):")
            for idx, row in tradeoff_df.head(10).iterrows():
                print(f"  {row['metric_1']} ↔ {row['metric_2']}")
                print(f"    Relationship Strength: {row['relationship_strength']:.4f} ({row['relationship_type']})")
                print(f"    Trade-off: {row['is_tradeoff']}\n")
        else:
            print(f"\n⚠ No negative correlations found in Cluster {cluster_id}")
            tradeoff_df = pd.DataFrame()
            if export_csv:
                tradeoff_df.to_csv(os.path.join(csv_dir, f'cluster_{cluster_id}_tradeoff_analysis.csv'), index=False)
            # Store empty list for this cluster
            tradeoff_analysis_per_cluster[cluster_id] = []
    
    return {
        'status': 'completed',
        'tradeoff_analysis_per_cluster': tradeoff_analysis_per_cluster
    }


# ============ PIPELINE STEPS - PHASE 1 CONTINUED: MODEL TRAINING & EVALUATION ============

def step_phase1_model_training_and_evaluation(results, pipeline, **kwargs):
    """
    Step 1.3-1.4: PHASE 1 - Model Training & SHAP Interpretation
    
    For each cluster:
    1. Use selected features from step_phase1_feature_selection
    2. Train XGBoost classifier for cluster vs. others
    3. Hyperparameter grid search with cross-validation
    4. Evaluate with AUC, balanced accuracy, confusion matrix, precision, recall, F1
    5. Generate local & global SHAP values using TreeExplainer
    
    Requires XGBoost to be installed: pip install xgboost
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score
    except ImportError:
        print("⚠ Warning: XGBoost not installed. Model training skipped.")
        print("  Install with: pip install xgboost")
        return {
            'status': 'skipped',
            'message': 'XGBoost not available',
            'model_results': []
        }
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    
    X_standardized = load_result.get('X_standardized')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    metric_cols = load_result.get('metric_cols')
    data_folder = load_result.get('data_folder', 'data')
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)

    selected_features_dict = feature_result.get('selected_features_per_cluster', {})
    
    if any(v is None for v in [X_standardized, cluster_labels, metric_cols]):
        raise KeyError("step_phase1_load_data: Required data not available for model training")
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 3: MODEL TRAINING & EVALUATION")
    print("Training XGBoost classifiers per cluster with SHAP interpretation")
    print("="*80)
    
    all_model_results = []
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            print(f"\n⊘ Cluster {cluster_id}: SKIPPED (small cluster)")
            continue
        
        cluster_mask = cluster_labels == cluster_id
        n_cluster = np.sum(cluster_mask)
        
        # Get selected features for this cluster
        selected_feat_str = selected_features_dict.get(cluster_id, '')
        if not selected_feat_str:
            print(f"\n⊘ Cluster {cluster_id}: No features selected, skipping model training")
            continue
        
        selected_features = [f.strip() for f in selected_feat_str.split(',') if f.strip()]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}: Training XGBoost Model")
        print(f"{'='*80}")
        print(f"Selected features ({len(selected_features)}): {selected_features[:5]}", end="")
        if len(selected_features) > 5:
            print(f" ... and {len(selected_features) - 5} more")
        else:
            print()
        
        # Create binary classification: this cluster vs. others
        y_binary = (cluster_labels == cluster_id).astype(int)
        
        # Get feature indices
        feature_indices = [metric_cols.index(f) for f in selected_features]
        X_selected = X_standardized[:, feature_indices]
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"Class distribution (train): {np.bincount(y_train)}")
        print(f"Class distribution (test): {np.bincount(y_test)}")
        
        # ========== Hyperparameter Grid Search ==========
        print(f"\nPerforming hyperparameter grid search with 5-fold cross-validation...")
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [30, 60],
            'eta': [0.15, 0.25],
        }
        
        xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='auc', verbosity=0)
        grid_search = GridSearchCV(xgb_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        print(f"✓ Best CV AUC: {best_cv_score:.4f}")
        print(f"  Best parameters: {best_params}")
        
        # ========== Model Evaluation ==========
        print(f"\nEvaluating on test set...")
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Test AUC: {auc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:")
        print(f"  Precision (cluster): {report['1']['precision']:.4f}")
        print(f"  Recall (cluster): {report['1']['recall']:.4f}")
        print(f"  F1 (cluster): {report['1']['f1-score']:.4f}")
        
        # ========== SHAP Interpretability ==========
        print(f"\nGenerating SHAP values for interpretation...")
        
        high_shap_features = []
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            # Handle SHAP output format (binary classification)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            
            # Global feature importance (mean |SHAP|)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importance = dict(zip(selected_features, mean_abs_shap))
            shap_importance_sorted = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
            
            # top_percentile = 95  # Top 5% of features
            # importance_values = [imp for _, imp in shap_importance_sorted]
            # threshold = np.percentile(importance_values, top_percentile) if importance_values else 0
            # high_shap_features = [feat for feat, importance in shap_importance_sorted if importance >= threshold]
            
            # Select all features with SHAP value > 0.1
            high_shap_features = [feat for feat, importance in shap_importance_sorted if importance > 0.1]
            
            print(f"✓ SHAP features with importance > 0.1 ({len(high_shap_features)} features):")
            for feat, importance in shap_importance_sorted:
                if importance > 0.1:
                    print(f"    {feat}: {importance:.6f}")
            
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {e}") 
            shap_importance_sorted = []
        
        # Store model results
        model_result = {
            'cluster_id': cluster_id,
            'n_samples': n_cluster,
            'n_features_used': len(selected_features),
            'selected_features': ','.join(selected_features),
            'test_auc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'balanced_accuracy': balanced_acc,
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
            'high_shap_features': ','.join(high_shap_features)
        }
        all_model_results.append(model_result)
        
       
    # Create summary results dataframe
    if all_model_results:
        models_summary = pd.DataFrame(all_model_results)
        if export_csv:
            models_file = os.path.join(csv_dir, 'workflows_model_evaluation_summary.csv')
            models_summary.to_csv(models_file, index=False)
            print(f"\n✓ Saved model evaluation summary to: {models_file}")
    else:
        models_summary = pd.DataFrame()
    
    return {
        'status': 'completed',
        'model_results': all_model_results,
        'models_summary': models_summary
    }


def step_phase1_decision_tree_rules(results, pipeline, **kwargs):
    """
    Step 1.5: Using imodels built-in rule evaluation
    """
    try:
        from imodels import SkopeRulesClassifier
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        import os
    except ImportError as e:
        print(f"⚠ Warning: {e}")
        return {'status': 'skipped', 'message': str(e)}
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    df_clustered = load_result.get('df_clustered')
    param_cols = load_result.get('param_cols')
    data_folder = load_result.get('data_folder', 'data')
    csv_dir = load_result.get('csv_dir', os.path.join(data_folder, 'csv'))
    export_csv = load_result.get('export_csv', True)
    
    if any(v is None for v in [cluster_labels, n_clusters, df_clustered, param_cols]):
        raise KeyError("Required data not available")
    
    # Sanitize column names
    hyperparameters = [col.replace(' ', '_') for col in param_cols]
    df_clustered.columns = [col.replace(' ', '_') for col in df_clustered.columns]
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 5: CLUSTER RULES (Using Built-in Rule Evaluation)")
    print("="*80)
    
    # Prepare data
    X_hyperparam = df_clustered[hyperparameters].copy()
    
    label_encoders = {}
    categorical_mappings = {}
    
    for col in X_hyperparam.columns:
        if X_hyperparam[col].dtype == 'object':
            le = LabelEncoder()
            X_hyperparam[col] = le.fit_transform(X_hyperparam[col])
            label_encoders[col] = le
            categorical_mappings[col] = {i: val for i, val in enumerate(le.classes_)}
    
    X_array = X_hyperparam.values
    y_cluster = cluster_labels
    
    def decode_categorical_in_rule(rule_str, categorical_mappings, label_encoders):
        """Decode categorical features"""
        import re
        decoded = rule_str
        
        for col, mapping in categorical_mappings.items():
            if col not in label_encoders:
                continue
            
            le = label_encoders[col]
            all_classes = le.classes_
            
            patterns = [
                (rf'{col}\s*<=\s*([\d.]+)', '<='),
                (rf'{col}\s*>\s*([\d.]+)', '>'),
            ]
            
            for pattern, operator in patterns:
                matches = list(re.finditer(pattern, decoded))
                for match in matches:
                    threshold = float(match.group(1))
                    original_condition = match.group(0)
                    
                    if operator == '<=':
                        valid_indices = [i for i in range(len(all_classes)) if i <= threshold]
                    else:
                        valid_indices = [i for i in range(len(all_classes)) if i > threshold]
                    
                    valid_categories = [all_classes[i] for i in valid_indices]
                    
                    if len(valid_categories) == 1:
                        replacement = f"{col} = '{valid_categories[0]}'"
                    else:
                        categories_str = ', '.join([f"'{c}'" for c in valid_categories])
                        replacement = f"{col} IN {{{categories_str}}}"
                    
                    decoded = decoded.replace(original_condition, replacement, 1)
        
        return decoded
    
    all_cluster_rules = {}
    cluster_rules_summary = []
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            continue
        
        y_binary = (y_cluster == cluster_id).astype(int)
        n_cluster_samples = y_binary.sum()
        
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id} ({n_cluster_samples} workflows)")
        print(f"{'─'*80}")
        
        clf = SkopeRulesClassifier(
            max_depth_duplication=3,
            n_estimators=30,
            precision_min=0.5,
            recall_min=0.1,
            max_depth=4,
            random_state=42
        )
        
        try:
            clf.fit(X_array, y_binary, feature_names=hyperparameters)
            
            rules_list = clf.rules_
            print(f"  ✓ Extracted {len(rules_list)} rules")
            
            if len(rules_list) == 0:
                continue
            
            # Get cluster-specific data
            cluster_mask = (y_cluster == cluster_id)
            X_cluster = X_array[cluster_mask]
            df_cluster = X_hyperparam[cluster_mask]
            
            cluster_rules = []
            for rule_obj in rules_list:
                rule_str = rule_obj.rule
                precision, recall, nb_samples_total = rule_obj.args
                
                # USE BUILT-IN: Query the rule on cluster data using pandas
                try:
                    # imodels internally uses df.query() - we do the same
                    satisfied_mask = df_cluster.query(rule_str).index
                    n_workflows_in_cluster = len(satisfied_mask)
                    
                    # Cluster-specific recall
                    cluster_recall = n_workflows_in_cluster / n_cluster_samples if n_cluster_samples > 0 else 0
                    
                except Exception as e:
                    print(f"    ⚠ Could not evaluate rule: {rule_str}")
                    continue
                
                decoded_rule = decode_categorical_in_rule(rule_str, categorical_mappings, label_encoders)
                
                if precision + cluster_recall > 0:
                    f1_score = 2 * (precision * cluster_recall) / (precision + cluster_recall)
                else:
                    f1_score = 0.0
                
                # Score based on cluster-specific counts
                # significance_weight = np.log1p(n_workflows_in_cluster)
                p_workflows_in_cluster = n_workflows_in_cluster / n_cluster_samples if n_cluster_samples > 0 else 0
                combined_score = f1_score * p_workflows_in_cluster
                
                cluster_rules.append({
                    'rule': decoded_rule,
                    'precision': precision,
                    'recall': cluster_recall,
                    'f1_score': f1_score,
                    'n_workflows_in_cluster': int(n_workflows_in_cluster),
                    'combined_score': combined_score
                })
            
            cluster_rules = sorted(cluster_rules, key=lambda x: x['combined_score'], reverse=True)
            all_cluster_rules[cluster_id] = cluster_rules
            
            print(f"\n  Top rules (by F1 × cluster coverage):")
            for idx, rule_info in enumerate(cluster_rules[:3], 1):
                print(f"\n  Rule {idx}: {rule_info['rule']}")
                print(f"           F1: {rule_info['f1_score']:.3f} | "
                      f"Precision: {rule_info['precision']:.3f} | "
                      f"Recall: {rule_info['recall']:.3f}")
                print(f"           Covers {rule_info['n_workflows_in_cluster']}/{n_cluster_samples} workflows in this cluster")
                
                cluster_rules_summary.append({
                    'cluster_id': cluster_id,
                    'rule_number': idx,
                    'rule': rule_info['rule'],
                    'precision': rule_info['precision'],
                    'recall': rule_info['recall'],
                    'f1_score': rule_info['f1_score'],
                    'n_workflows_in_cluster': rule_info['n_workflows_in_cluster'],
                    'cluster_size': n_cluster_samples,
                    'combined_score': rule_info['combined_score']
                })
        
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if cluster_rules_summary:
        rules_df = pd.DataFrame(cluster_rules_summary)
        if export_csv:
            rules_file = os.path.join(csv_dir, 'cluster_decision_rules.csv')
            rules_df.to_csv(rules_file, index=False)
            print(f"\n✓ Saved to: {rules_file}")
    else:
        rules_df = pd.DataFrame()
    
    print("\n" + "="*80)
    print("BEST RULE PER CLUSTER")
    print("="*80)
    for cluster_id in sorted(all_cluster_rules.keys()):
        rules = all_cluster_rules[cluster_id]
        if rules:
            main_rule = rules[0]
            n_total = df_clustered[df_clustered['cluster'] == cluster_id].shape[0]
            print(f"\nCluster {cluster_id}: {main_rule['rule']}")
            print(f"  [F1: {main_rule['f1_score']:.3f}, covers {main_rule['n_workflows_in_cluster']}/{n_total} workflows]")
    
    return {
        'status': 'completed',
        'cluster_rules': all_cluster_rules,
        'rules_summary': rules_df
    }



def step_phase1_comprehensive_cluster_insights(results, pipeline, **kwargs):
    """
    Step 1.6: Generate Comprehensive Cluster Statistics JSON
    
    Aggregates all important statistics from previous steps into a single
    comprehensive JSON file for each cluster containing:
    - Cluster metadata (size, samples)
    - Feature selection results (n_selected, selected_features)
    - Model evaluation metrics (AUC, F1, balanced accuracy, precision, recall)
    - High SHAP features (features with importance > 0.1)
    - Trade-off analysis (negative correlations between selected and non-selected)
    - Hyperparameter patterns (dominant values and percentages)
    - Decision tree rules (interpretable hyperparameter combinations)
    """
    import json
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    feature_result = pipeline.get_result('step_phase1_feature_selection', default={})
    model_result = pipeline.get_result('step_phase1_model_training', default={})
    rules_result = pipeline.get_result('step_phase1_decision_tree_rules', default={})
    tradeoff_result = pipeline.get_result('step_phase1_tradeoff_analysis', default={})
    
    df_clustered = load_result.get('df_clustered')
    medoids = load_result.get('medoids')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    metric_cols = load_result.get('metric_cols')
    param_cols = load_result.get('param_cols')
    data_folder = load_result.get('data_folder', 'data')
    X_standardized = load_result.get('X_standardized')
    correlation_threshold = kwargs.get('correlation_threshold', 0.75)

    results_summary = feature_result.get('results_summary', pd.DataFrame())
    models_summary = model_result.get('models_summary', pd.DataFrame())
    rules_summary = rules_result.get('rules_summary', pd.DataFrame())
    
    if df_clustered is None:
        raise KeyError("step_phase1_load_data: Required data not available")
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 6: GENERATING COMPREHENSIVE CLUSTER STATISTICS")
    print("="*80)
    
    # Use loaded hyperparameters
    hyperparameters = param_cols
    
    cluster_insights_dict = {}
    
    for cluster_id in range(n_clusters):
        if cluster_id in small_clusters:
            continue
        
        cluster_mask = cluster_labels == cluster_id
        cluster_df = df_clustered[cluster_mask].copy()
        n_cluster = len(cluster_df)
        
        # Get medoid ID for this cluster
        medoid_id = None
        medoid_index = None
        if medoids is not None and not medoids.empty:
            cluster_medoids = medoids[medoids['cluster_id'] == cluster_id]
            if not cluster_medoids.empty:
                medoid_id = cluster_medoids.iloc[0]['workflow_id'] if 'workflow_id' in cluster_medoids.columns else None
                medoid_index = int(cluster_medoids.iloc[0]['medoid_index']) if 'medoid_index' in cluster_medoids.columns else None
        
        cluster_insights = {
            'cluster_id': cluster_id,
            'metadata': {
                'n_workflows': n_cluster,
                'percentage_of_total': round((n_cluster / len(df_clustered)) * 100, 2),
                'medoid_workflow_id': medoid_id,
                'medoid_index': medoid_index,
            },
            'feature_selection': {},
            'model_evaluation': {},
            'high_shap_features': [],
            'correlation_analysis': {},
            'trade_off_analysis': {},
            'hyperparameter_patterns': {},
            'decision_tree_rules': []
        }
        
        # ===== FEATURE SELECTION INSIGHTS =====
        if not results_summary.empty:
            cluster_fs = results_summary[results_summary['cluster_id'] == cluster_id]
            if not cluster_fs.empty:
                row = cluster_fs.iloc[0]
                selected_feats = row['selected_features'].split(',') if isinstance(row['selected_features'], str) else []
                cluster_insights['feature_selection'] = {
                    'n_features_selected': row['n_features_selected'],
                    'selected_features': selected_feats,
                    'n_metrics_total': len(metric_cols)
                }
                
                # Compute feature statistics comparing cluster vs others
                # This will be reused for high_shap_features to avoid redundant computation
                feature_stats = {}
                other_mask = cluster_labels != cluster_id
                
                for feat in selected_feats:
                    if feat in metric_cols:
                        feat_idx = metric_cols.index(feat)
                        
                        # Get values for this cluster
                        cluster_values = X_standardized[cluster_mask, feat_idx]
                        cluster_mean = float(np.mean(cluster_values))
                        cluster_std = float(np.std(cluster_values))
                        
                        # Get values for all other clusters
                        other_values = X_standardized[other_mask, feat_idx]
                        other_mean = float(np.mean(other_values))
                        other_std = float(np.std(other_values))
                        
                        # Compute global stats for z-score
                        all_values = X_standardized[:, feat_idx]
                        global_mean = float(np.mean(all_values))
                        global_std = float(np.std(all_values))
                        
                        # Classify cluster's mean value
                        z_score = (cluster_mean - global_mean) / (global_std + 1e-6)
                        threshold = 1.0
                        if z_score > threshold:
                            value_category = "high"  # > threshold std above global
                        elif z_score < -threshold:
                            value_category = "low"   # < threshold std below global
                        else:
                            value_category = "mid"
                        
                        # Calculate how distinctive this feature is for the cluster
                        distinctiveness = abs(cluster_mean - other_mean) / (other_std + 1e-6)
                        
                        feature_stats[feat] = {
                            'cluster_mean': round(cluster_mean, 4),
                            'cluster_std': round(cluster_std, 4),
                            'other_clusters_mean': round(other_mean, 4),
                            'other_clusters_std': round(other_std, 4),
                            'value_category': value_category,
                            'distinctiveness_score': round(distinctiveness, 4),
                            'z-score': round(z_score, 4),
                        }
                
                cluster_insights['feature_selection']['feature_statistics'] = feature_stats
        
        # ===== CORRELATION ANALYSIS FOR REMOVED FEATURES =====
        # Get removed features analysis from feature selection step
        removed_features_per_cluster = feature_result.get('removed_features_analysis_per_cluster', {})
        cluster_removed_features = removed_features_per_cluster.get(cluster_id, {})
        
        if cluster_removed_features:
            cluster_insights['correlation_analysis'] = {
                'n_removed_features': len(cluster_removed_features),
                'removed_features': {
                    feat: {
                        'max_relationship': round(float(details['max_relationship']), 4),
                        'related_to': details['related_to'],
                        'all_relationships': details.get('all_relationships', [])
                    }
                    for feat, details in cluster_removed_features.items()
                }
            }
        
        # ===== MODEL EVALUATION INSIGHTS =====
        if not models_summary.empty:
            cluster_model = models_summary[models_summary['cluster_id'] == cluster_id]
            if not cluster_model.empty:
                row = cluster_model.iloc[0]
                high_shap = row['high_shap_features'].split(',') if isinstance(row['high_shap_features'], str) else []
                high_shap = [f.strip() for f in high_shap if f.strip()]
                
                # Calculate model quality score
                test_auc = float(row['test_auc'])
                f1 = float(row['f1_score'])
                precision = float(row['precision'])
                recall = float(row['recall'])
                balanced_acc = float(row['balanced_accuracy']) if 'balanced_accuracy' in row else None
                
                # Quality score: weighted combination of metrics
                # Prioritize AUC as main discriminator, then balanced F1
                model_quality_score = round((test_auc * 0.6 + f1 * 0.4), 4)
                
                # Generate quality interpretation
                if model_quality_score >= 0.9:
                    quality_level = "Excellent - Cluster is very well distinguished"
                elif model_quality_score >= 0.7:
                    quality_level = "Good - Cluster is well distinguished"
                elif model_quality_score >= 0.5:
                    quality_level = "Fair - Cluster has moderate distinction"
                else:
                    quality_level = "Poor - Cluster is not well distinguished"
                
                cluster_insights['model_evaluation'] = {
                    'test_auc': round(float(row['test_auc']), 4),
                    'balanced_accuracy': round(float(balanced_acc), 4) if balanced_acc is not None else None,
                    'precision': round(float(row['precision']), 4),
                    'recall': round(float(row['recall']), 4),
                    'f1_score': round(float(row['f1_score']), 4),
                    'model_quality_score': model_quality_score,
                    'quality_interpretation': quality_level,
                    'confusion_matrix': {
                        'true_negatives': int(row['tn']),
                        'false_positives': int(row['fp']),
                        'false_negatives': int(row['fn']),
                        'true_positives': int(row['tp'])
                    }
                }
                
                # Reuse already computed feature_stats for high SHAP features
                # High SHAP features are a subset of selected features
                shap_feature_stats = {feat: feature_stats[feat] for feat in high_shap if feat in feature_stats}
                
                cluster_insights['high_shap_features'] = {
                    'features': list(shap_feature_stats.keys()),
                    'feature_statistics': shap_feature_stats
                }
        
        # ===== TRADE-OFF ANALYSIS =====
        # Get trade-off data from trade-off analysis step
        tradeoff_data_per_cluster = tradeoff_result.get('tradeoff_analysis_per_cluster', {})
        cluster_tradeoffs = tradeoff_data_per_cluster.get(cluster_id, [])
        
        if cluster_tradeoffs:
            # Convert to list of dicts with rounded values
            strong_tradeoffs = []
            
            for tradeoff in cluster_tradeoffs:
                # Round numeric values
                if(tradeoff["is_tradeoff"] == 1):
                    strong_tradeoffs.append(tradeoff)
            
            cluster_insights['trade_off_analysis'] = {
                'n_total_tradeoffs': len(cluster_tradeoffs),
                'n_strong_tradeoffs': len(strong_tradeoffs),
                'strong_threshold': -correlation_threshold,
                'strong_tradeoffs': strong_tradeoffs
            }
        
        # ===== HYPERPARAMETER PATTERNS =====
        for hyperparam in hyperparameters:
            if hyperparam in cluster_df.columns:
                value_counts = cluster_df[hyperparam].value_counts()
                dominant_value = value_counts.index[0]
                dominant_pct = round((value_counts.iloc[0] / n_cluster) * 100, 2)
                
                cluster_insights['hyperparameter_patterns'][hyperparam] = {
                    'dominant_value': str(dominant_value),
                    'dominant_percentage': float(dominant_pct),
                    'unique_values': int(value_counts.shape[0]),
                    'value_distribution': {str(k): int(v) for k, v in value_counts.items()}
                }
        
        # ===== DECISION TREE RULES WITH SCORES =====
        if not rules_summary.empty:
            cluster_rules_filtered = rules_summary[rules_summary['cluster_id'] == cluster_id]
            if not cluster_rules_filtered.empty:
                # Include rules with their scores
                rules_with_scores = []
                for idx, row in cluster_rules_filtered.iterrows():
                    rules_with_scores.append({
                        'rule': row['rule'],
                        'f1_score': float(row['f1_score']),
                        'precision': float(row['precision']),
                        'recall': float(row['recall']),
                        'n_workflows_in_cluster': int(row['n_workflows_in_cluster']),
                        'combined_score': float(row['combined_score']) if 'combined_score' in row and pd.notna(row['combined_score']) else None
                    })
                cluster_insights['decision_tree_rules'] = rules_with_scores
        
        cluster_insights_dict[str(cluster_id)] = cluster_insights
    
    # ===== SAVE COMPREHENSIVE JSON =====
    json_file = os.path.join(data_folder, 'clusters_comprehensive_insights.json')
    with open(json_file, 'w') as f:
        json.dump(cluster_insights_dict, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Saved comprehensive cluster statistics to: {json_file}")
    print(f"  Clusters included: {list(cluster_insights_dict.keys())}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTER STATISTICS SUMMARY")
    print("="*80)
    for cluster_id, insights in cluster_insights_dict.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Workflows: {insights['metadata']['n_workflows']} ({insights['metadata']['percentage_of_total']}%)")
        if insights['feature_selection']:
            n_selected = insights['feature_selection']['n_features_selected']
            print(f"  Selected Features: {n_selected}")
            # Show representative features with their categories
            if 'feature_statistics' in insights['feature_selection']:
                print(f"    Representative features (low/mid/high):")
                feat_stats = insights['feature_selection']['feature_statistics']
                for feat, stats in list(feat_stats.items())[:3]:
                    category = stats['value_category']
                    dist_score = stats['distinctiveness_score']
                    print(f"      - {feat}: {category.upper()} (distinctiveness: {dist_score:.2f})")
        if insights['correlation_analysis'] and 'n_removed_features' in insights['correlation_analysis']:
            print(f"  Correlation Analysis: {insights['correlation_analysis']['n_removed_features']} removed features analyzed")
        if insights['model_evaluation']:
            print(f"  Quality: {insights['model_evaluation']['quality_interpretation']}")
        if insights['high_shap_features']:
            features_list = insights['high_shap_features'] if isinstance(insights['high_shap_features'], list) else insights['high_shap_features'].get('features', [])
            n_features = len(features_list)
            print(f"  High SHAP Features: {n_features}")
            # Show SHAP features with their categories
            if isinstance(insights['high_shap_features'], dict) and 'feature_statistics' in insights['high_shap_features']:
                print(f"    Key discriminative features (low/mid/high):")
                shap_stats = insights['high_shap_features']['feature_statistics']
                for feat, stats in list(shap_stats.items())[:3]:
                    category = stats['value_category']
                    dist_score = stats['distinctiveness_score']
                    print(f"      - {feat}: {category.upper()} (distinctiveness: {dist_score:.2f})")
        if insights['trade_off_analysis'] and 'n_strong_tradeoffs' in insights['trade_off_analysis']:
            print(f"  Strong Trade-offs: {insights['trade_off_analysis']['n_strong_tradeoffs']}")
        if insights["decision_tree_rules"]:
            print(f"  Decision Tree Rules: {len(insights['decision_tree_rules'])}")
            print(f"    Top Rule: {insights['decision_tree_rules'][0]['rule']}")
    
    return {
        'status': 'completed',
        'cluster_insights': cluster_insights_dict,
        'json_file': json_file
    }


def build_default_insights_pipeline():
    """Build the default insights pipeline with all steps."""
    pipeline = InsightsPipeline()
    
    # PHASE 1: Feature Selection & Model Training on Original Data
    pipeline.add_step('step_phase1_load_data', step_phase1_load_data, enabled=True)
    pipeline.add_step('step_phase1_feature_selection', step_phase1_feature_selection, enabled=True)
    pipeline.add_step('step_phase1_tradeoff_analysis', step_phase1_tradeoff_analysis, enabled=True)
    pipeline.add_step('step_phase1_model_training', step_phase1_model_training_and_evaluation, enabled=True)
    pipeline.add_step('step_phase1_decision_tree_rules', step_phase1_decision_tree_rules, enabled=True)
    pipeline.add_step('step_phase1_comprehensive_cluster_insights', step_phase1_comprehensive_cluster_insights, enabled=True)
    
    return pipeline


if __name__ == "__main__":
    main()
