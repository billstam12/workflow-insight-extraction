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


def load_clustering_results(prefix='workflows'):
    """Load clustering output files."""
    df_clustered = pd.read_csv(f'{prefix}_clustered.csv')
    medoids = pd.read_csv(f'{prefix}_medoids.csv')
    
    # Load cluster metadata to identify small/outlier clusters
    try:
        cluster_metadata = pd.read_csv(f'{prefix}_cluster_metadata.csv')
    except FileNotFoundError:
        cluster_metadata = None

    print(f"Loaded clustering results:")
    print(f"  - {len(df_clustered)} workflows in {df_clustered['cluster'].nunique()} clusters")
    
    if cluster_metadata is not None:
        small_clusters = cluster_metadata[cluster_metadata['is_small'] == True]['cluster_id'].tolist()
        if small_clusters:
            print(f"  - Small/outlier clusters detected: {small_clusters}")

    return df_clustered, medoids, cluster_metadata


def load_processed_data(prefix='workflows'):
    """Load pre-processed (dimensionality reduced) data for Phase 2."""
    try:
        X_processed_df = pd.read_csv(f'{prefix}_processed_data.csv')
        print(f"Loaded processed data: {X_processed_df.shape}")
        return X_processed_df
    except FileNotFoundError:
        print(f"⚠ Processed data file not found: {prefix}_processed_data.csv")
        return None


def identify_metric_columns(df):
    """Identify metric columns (excluding hyperparameters and system metrics)."""
    exclude_cols = ['workflowId', 'cluster', 'criterion', 'fairness method', 
                    'random state', 'max depth', 'normalization', 'n estimators']
    metrics = [col for col in df.columns if col not in exclude_cols]
    return metrics


def feature_selection_shap_iterative(X_train, y_train, feature_names, n_iterations=None, correlation_threshold=0.9):
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
    
    return selected_features, selection_history


def identify_metric_columns(df):
    """Identify metric columns (excluding hyperparameters and system metrics)."""
    exclude_cols = ['workflowId', 'cluster', 'criterion', 'fairness method', 
                    'random state', 'max depth', 'normalization', 'n estimators']
    metrics = [col for col in df.columns if col not in exclude_cols]
    return metrics


def main():
    """Main execution function """
    print("="*80)
    print("ML WORKFLOW CLASSIFICATION & INTERPRETABILITY")
    print("  PHASE 1: Feature Selection on Original Data (all features)")
    print("  PHASE 2: Model Training on Selected Features (future: with PCA-clustered data)")
    print("="*80 + "\n")

    # Load clustering results and processed data
    df_clustered, medoids, cluster_metadata = load_clustering_results()
    X_processed_df = load_processed_data()

    # Identify metric columns from original data
    metric_cols = identify_metric_columns(df_clustered)
    print(f"Total features to analyze: {len(metric_cols)}")
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
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'small_clusters': small_clusters,
        'correlation_threshold': 0.5,
        'n_iterations': None
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
        print(model_summary[['cluster_id', 'n_samples', 'best_cv_auc', 'test_auc', 'f1_score']].to_string(index=False))

    print("\n" + "="*80)
    print("Analysis Complete! Generated Files:")
    print("="*80)
    print("  PHASE 1 - Feature Selection:")
    print("    - workflows_classification_results.csv (per-cluster summary)")
    print("    - cluster_X_selection_history.csv (feature removal tracking)")
    print("    - cluster_X_tradeoff_analysis.csv (negative correlations/trade-offs)")
    print("  PHASE 1 - Model Training:")
    print("    - workflows_model_evaluation_summary.csv (per-cluster model performance)")
    print("  PHASE 1 - Decision Tree Rules:")
    print("    - workflows_decision_tree_rules.csv (interpretable decision paths per workflow)")
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
    cluster_labels = kwargs.get('cluster_labels')
    n_clusters = kwargs.get('n_clusters')
    small_clusters = kwargs.get('small_clusters', set())
    
    return {
        'df_clustered': df_clustered,
        'medoids': medoids,
        'X_standardized': X_standardized,
        'X_processed_df': X_processed_df,
        'metric_cols': metric_cols,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'small_clusters': small_clusters
    }


def step_phase1_feature_selection(results, pipeline, **kwargs):
    """
    Step 1.2-1.3: PHASE 1 - Feature Selection on Original Data
    
    Per-cluster feature selection:
    - Train-test split (80-20)
    - Multi-step feature selection (correlation, SHAP-based)
    - Track removed features and trade-offs
    """
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    df_clustered = load_result.get('df_clustered')
    X_standardized = load_result.get('X_standardized')
    metric_cols = load_result.get('metric_cols')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    
    correlation_threshold = kwargs.get('correlation_threshold', 0.5)
    n_iterations = kwargs.get('n_iterations', None)
    
    if any(v is None for v in [X_standardized, metric_cols, cluster_labels]):
        raise KeyError("step_phase1_load_data: Required data not loaded")
    
    print("\n" + "="*80)
    print("PHASE 1: FEATURE SELECTION ON ORIGINAL DATA (All Features - Standardized)")
    print("="*80)
    
    all_results = []
    
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
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_standardized, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
        print(f"Class distribution (train): {np.bincount(y_train)}")
        print(f"Class distribution (test): {np.bincount(y_test)}")
        
        # Feature selection (Steps 1-3 from paper)
        selected_features, selection_history = feature_selection_shap_iterative(
            X_train, y_train, metric_cols, n_iterations=n_iterations, 
            correlation_threshold=correlation_threshold
        )
        
        print(f"\n✓ Final selected features ({len(selected_features)}): {selected_features}")
        
        # Store results
        cluster_result = {
            'cluster_id': cluster_id,
            'n_samples': n_cluster,
            'n_features_selected': len(selected_features),
            'selected_features': ','.join(selected_features),
        }
        all_results.append(cluster_result)
        
        # Save selection history
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
            selection_df.to_csv(f'cluster_{cluster_id}_selection_history.csv', index=False)
            print(f"✓ Saved cluster_{cluster_id}_selection_history.csv ({len(selection_df)} removed features tracked)")
        
        # Save trade-off analysis - SELECTED vs NON-SELECTED FEATURES
        print("\n" + "-"*80)
        print(f"Computing Trade-off Metrics: Selected vs Non-Selected Features")
        print("-"*80)
        
        cluster_mask = cluster_labels == cluster_id
        X_cluster = X_standardized[cluster_mask]
        X_cluster_df = pd.DataFrame(X_cluster, columns=metric_cols)
        
        var_types = detect_variable_types(X_cluster_df)
        correlation_results = []
        
        # Separate selected and non-selected features
        selected_features_list = selected_features if isinstance(selected_features, list) else []
        non_selected_features = [f for f in metric_cols if f not in selected_features_list]
        
        print(f"Selected features: {len(selected_features_list)}")
        print(f"Non-selected features: {len(non_selected_features)}")
        print(f"Analyzing pairs: selected vs non-selected (excluding non-selected vs non-selected)")
        
        # Analyze pairs: selected vs non-selected
        for feat1 in selected_features_list:
            for feat2 in non_selected_features:
                try:
                    measure, measure_type = compute_relationship_measure(
                        X_cluster_df, feat1, feat2, var_types
                    )
                    
                    x = X_cluster_df[feat1].values
                    y = X_cluster_df[feat2].values
                    actual_corr = np.corrcoef(x, y)[0, 1]
                    
                    # Only include negative relationships as trade-offs
                    if actual_corr < 0:
                        correlation_results.append({
                            'metric_1': feat1,
                            'metric_2': feat2,
                            'relationship_type': measure_type,
                            'relationship_strength': measure,
                            'actual_correlation': actual_corr,
                            'is_tradeoff': 'Yes' if actual_corr < -0.9 else 'Weak'
                        })
                except Exception as e:
                    pass
        
        if correlation_results:
            tradeoff_df = pd.DataFrame(correlation_results)
            tradeoff_df = tradeoff_df.sort_values('relationship_strength', ascending=False)
            tradeoff_df.to_csv(f'cluster_{cluster_id}_tradeoff_analysis.csv', index=False)
            
            print(f"\n✓ Found {len(tradeoff_df)} negative correlations (trade-offs) in Cluster {cluster_id}")
            print(f"✓ Saved cluster_{cluster_id}_tradeoff_analysis.csv")
            
            print("\nTop Trade-off Relationships (Strongest Negative Correlations):")
            for idx, row in tradeoff_df.head(10).iterrows():
                print(f"  {row['metric_1']} ↔ {row['metric_2']}")
                print(f"    Actual Correlation: {row['actual_correlation']:.4f}")
                print(f"    Relationship Strength: {row['relationship_strength']:.4f} ({row['relationship_type']})")
                print(f"    Trade-off: {row['is_tradeoff']}\n")
        else:
            print(f"\n⚠ No negative correlations found in Cluster {cluster_id}")
            tradeoff_df = pd.DataFrame()
            tradeoff_df.to_csv(f'cluster_{cluster_id}_tradeoff_analysis.csv', index=False)
    
    # Create summary results dataframe
    if all_results:
        results_summary = pd.DataFrame(all_results)
        results_file = 'workflows_classification_results.csv'
        results_summary.to_csv(results_file, index=False)
        print(f"\n✓ Saved classification results summary to: {results_file}")
    else:
        results_summary = pd.DataFrame()
    
    return {
        'results_summary': results_summary,
        'selected_features_per_cluster': {r['cluster_id']: r['selected_features'] for r in all_results}
    }


# ============ PIPELINE STEPS - PHASE 1 CONTINUED: MODEL TRAINING & EVALUATION ============

def step_phase1_model_training_and_evaluation(results, pipeline, **kwargs):
    """
    Step 1.3-1.4: PHASE 1 - Model Training & SHAP Interpretation
    
    For each cluster:
    1. Use selected features from step_phase1_feature_selection
    2. Train XGBoost classifier for cluster vs. others
    3. Hyperparameter grid search with cross-validation
    4. Evaluate with AUC, confusion matrix, precision, recall, F1
    5. Generate local & global SHAP values using TreeExplainer
    
    Requires XGBoost to be installed: pip install xgboost
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
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
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
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
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Test AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:")
        print(f"  Precision (cluster): {report['1']['precision']:.4f}")
        print(f"  Recall (cluster): {report['1']['recall']:.4f}")
        print(f"  F1 (cluster): {report['1']['f1-score']:.4f}")
        
        # ========== SHAP Interpretability ==========
        print(f"\nGenerating SHAP values for interpretation...")
        
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
            'best_cv_auc': best_cv_score,
            'test_auc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
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
        models_file = 'workflows_model_evaluation_summary.csv'
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
    Step 1.5: Decision Tree Rules for Workflows
    
    For each cluster:
    1. Train a shallow decision tree on HYPERPARAMETERS (structural components)
    2. Extract general interpretable rules for each CLUSTER (not individual workflows)
    3. Generate human-readable rules explaining what defines cluster membership
    
    Output: General rules like:
    "Cluster 0: IF max_depth > 11.5 AND fairness_method IN {'none', 'disparate_impact_remover'}"
    
    This answers: "What hyperparameter combination defines this cluster?"
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, _tree
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
    except ImportError:
        print("⚠ Warning: sklearn not available for decision tree rules.")
        return {'status': 'skipped', 'message': 'sklearn not available'}
    
    load_result = pipeline.get_result('step_phase1_load_data', default={})
    
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    df_clustered = load_result.get('df_clustered')
    
    if any(v is None for v in [cluster_labels, n_clusters, df_clustered]):
        raise KeyError("step_phase1_load_data: Required data not available for decision tree rules")
    
    # Define hyperparameters (structural components)
    hyperparameters = ['criterion', 'fairness method', 'random state', 
                       'max depth', 'normalization', 'n estimators']
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 5: DECISION TREE RULES FOR CLUSTERS (HYPERPARAMETER-BASED)")
    print("Extracting general decision rules that define each cluster")
    print("="*80)
    
    le_dict = {}  # Store label encoders for decoding
    
    # Prepare hyperparameter data
    X_hyperparam = df_clustered[hyperparameters].copy()
    
    # Encode categorical hyperparameters
    for col in X_hyperparam.columns:
        if X_hyperparam[col].dtype == 'object':
            le = LabelEncoder()
            X_hyperparam[col] = le.fit_transform(X_hyperparam[col])
            le_dict[col] = le
    
    X_hyperparam_array = X_hyperparam.values
    y_cluster = cluster_labels
    
    # Train a decision tree on hyperparameters to predict cluster
    print(f"\nTraining Decision Tree on {len(hyperparameters)} hyperparameters to predict cluster membership...")
    
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=max(2, len(df_clustered) // 50),
        random_state=42
    )
    dt.fit(X_hyperparam_array, y_cluster)
    
    print(f"Decision Tree trained with max_depth=5")
    print(f"Tree Accuracy on all data: {dt.score(X_hyperparam_array, y_cluster):.4f}")
    
    # Helper function to extract all paths to leaf nodes (general cluster rules)
    def extract_cluster_rules(tree, feature_names, le_dict):
        """
        Extract general rules for each cluster by traversing to leaf nodes.
        Returns one rule per cluster (or multiple if cluster appears in multiple leaves).
        """
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        
        cluster_rules = {}
        
        def recurse(node, path_conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node - split point
                feature_idx = tree_.feature[node]
                threshold = tree_.threshold[node]
                feature = feature_name[node]
                
                # Left branch (<=)
                left_conditions = path_conditions.copy()
                if feature in le_dict:
                    le = le_dict[feature]
                    valid_encoded = [i for i in range(len(le.classes_)) if i <= threshold]
                    valid_values = [le.classes_[i] for i in valid_encoded]
                    if len(valid_values) == 1:
                        left_conditions.append(f"{feature} = '{valid_values[0]}'")
                    else:
                        values_str = ', '.join([f"'{v}'" for v in valid_values])
                        left_conditions.append(f"{feature} IN {{{values_str}}}")
                else:
                    left_conditions.append(f"{feature} ≤ {threshold:.2f}")
                
                recurse(tree_.children_left[node], left_conditions)
                
                # Right branch (>)
                right_conditions = path_conditions.copy()
                if feature in le_dict:
                    le = le_dict[feature]
                    valid_encoded = [i for i in range(len(le.classes_)) if i > threshold]
                    valid_values = [le.classes_[i] for i in valid_encoded]
                    if len(valid_values) == 1:
                        right_conditions.append(f"{feature} = '{valid_values[0]}'")
                    else:
                        values_str = ', '.join([f"'{v}'" for v in valid_values])
                        right_conditions.append(f"{feature} IN {{{values_str}}}")
                else:
                    right_conditions.append(f"{feature} > {threshold:.2f}")
                
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf node - extract cluster and rule
                cluster_id = int(np.argmax(tree_.value[node][0]))
                n_samples = int(tree_.n_node_samples[node])
                rule_str = " AND ".join(path_conditions) if path_conditions else "All workflows"
                
                if cluster_id not in cluster_rules:
                    cluster_rules[cluster_id] = []
                cluster_rules[cluster_id].append({
                    'rule': rule_str,
                    'n_samples': n_samples
                })
        
        recurse(0, [])
        return cluster_rules
    
    # Extract general cluster rules
    cluster_rules = extract_cluster_rules(dt, hyperparameters, le_dict)
    
    # Display cluster rules
    print("\n" + "="*80)
    print("GENERAL CLUSTER RULES")
    print("="*80)
    
    cluster_rules_summary = []
    for cluster_id in sorted(cluster_rules.keys()):
        if cluster_id in small_clusters:
            continue
        
        rules = cluster_rules[cluster_id]
        n_workflows = df_clustered[df_clustered['cluster'] == cluster_id].shape[0]
        
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id}: {n_workflows} workflows")
        print(f"{'─'*80}")
        
        for idx, rule_info in enumerate(rules, 1):
            rule = rule_info['rule']
            n_samples = rule_info['n_samples']
            
            print(f"\nRule {idx}: IF {rule}")
            print(f"         THEN → Cluster {cluster_id}")
            print(f"         ({n_samples} workflows match this rule)")
            
            cluster_rules_summary.append({
                'cluster_id': cluster_id,
                'rule_number': idx,
                'rule': rule,
                'n_workflows': n_samples
            })
    
    # Save cluster rules to CSV
    if cluster_rules_summary:
        rules_df = pd.DataFrame(cluster_rules_summary)
        rules_file = 'cluster_decision_rules.csv'
        rules_df.to_csv(rules_file, index=False)
        print(f"\n✓ Saved cluster decision rules to: {rules_file}")
        print(f"  Total clusters with rules: {len(cluster_rules)}")
    else:
        rules_df = pd.DataFrame()
    
    # Also create a simple summary
    print("\n" + "="*80)
    print("CLUSTER SIGNATURE SUMMARY")
    print("="*80)
    for cluster_id in sorted(cluster_rules.keys()):
        if cluster_id in small_clusters:
            continue
        rules = cluster_rules[cluster_id]
        main_rule = rules[0]['rule']  # Primary rule
        n_workflows = df_clustered[df_clustered['cluster'] == cluster_id].shape[0]
        print(f"Cluster {cluster_id} ({n_workflows} workflows): {main_rule}")
    
    return {
        'status': 'completed',
        'cluster_rules': cluster_rules,
        'rules_summary': rules_df
    }


def step_phase1_comprehensive_cluster_insights(results, pipeline, **kwargs):
    """
    Step 1.6: Generate Comprehensive Cluster Statistics JSON
    
    Aggregates all important statistics from previous steps into a single
    comprehensive JSON file for each cluster containing:
    - Cluster metadata (size, samples)
    - Feature selection results (n_selected, selected_features)
    - Model evaluation metrics (AUC, F1, precision, recall)
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
    
    df_clustered = load_result.get('df_clustered')
    medoids = load_result.get('medoids')
    cluster_labels = load_result.get('cluster_labels')
    n_clusters = load_result.get('n_clusters')
    small_clusters = load_result.get('small_clusters', set())
    metric_cols = load_result.get('metric_cols')
    
    results_summary = feature_result.get('results_summary', pd.DataFrame())
    models_summary = model_result.get('models_summary', pd.DataFrame())
    rules_summary = rules_result.get('rules_summary', pd.DataFrame())
    
    if df_clustered is None:
        raise KeyError("step_phase1_load_data: Required data not available")
    
    print("\n" + "="*80)
    print("PHASE 1 STEP 6: GENERATING COMPREHENSIVE CLUSTER STATISTICS")
    print("="*80)
    
    # Define hyperparameters
    hyperparameters = ['criterion', 'fairness method', 'random state', 
                       'max depth', 'normalization', 'n estimators']
    
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
                    'best_cv_auc': round(float(row['best_cv_auc']), 4),
                    'test_auc': round(float(row['test_auc']), 4),
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
                
                cluster_insights['high_shap_features'] = high_shap
        
        # ===== TRADE-OFF ANALYSIS =====
        try:
            tradeoff_file = f'cluster_{cluster_id}_tradeoff_analysis.csv'
            tradeoff_df = pd.read_csv(tradeoff_file)
            if not tradeoff_df.empty:
                # Filter to only strong trade-offs (correlation < -0.9)
                strong_tradeoffs = tradeoff_df[tradeoff_df['actual_correlation'] < -0.9]
                
                cluster_insights['trade_off_analysis'] = {
                    'n_strong_tradeoffs': len(strong_tradeoffs),
                    'top_tradeoffs': []
                }
                
                for idx, row in strong_tradeoffs.head(5).iterrows():
                    cluster_insights['trade_off_analysis']['top_tradeoffs'].append({
                        'metric_1': row['metric_1'],
                        'metric_2': row['metric_2'],
                        'correlation': round(float(row['actual_correlation']), 4),
                        'relationship_strength': round(float(row['relationship_strength']), 4)
                    })
        except FileNotFoundError:
            pass
        
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
        
        # ===== DECISION TREE RULES =====
        if not rules_summary.empty:
            cluster_rules = rules_summary[rules_summary['cluster_id'] == cluster_id]
            if not cluster_rules.empty:
                # Get unique rules (sample) - using 'rule' column instead
                unique_rules = cluster_rules['rule'].unique()[:5] if 'rule' in cluster_rules.columns else []
                cluster_insights['decision_tree_rules'] = list(unique_rules)
        
        cluster_insights_dict[str(cluster_id)] = cluster_insights
    
    # ===== SAVE COMPREHENSIVE JSON =====
    json_file = 'clusters_comprehensive_insights.json'
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
            print(f"  Selected Features: {insights['feature_selection']['n_features_selected']}")
        if insights['model_evaluation']:
            print(f"  Model AUC (test): {insights['model_evaluation']['test_auc']}")
            print(f"  F1 Score: {insights['model_evaluation']['f1_score']}")
            print(f"  Model Quality Score: {insights['model_evaluation']['model_quality_score']}")
            print(f"  Quality: {insights['model_evaluation']['quality_interpretation']}")
        if insights['high_shap_features']:
            print(f"  High SHAP Features ({len(insights['high_shap_features'])}): {', '.join(insights['high_shap_features'][:3])}...")
        if insights['trade_off_analysis'] and 'n_strong_tradeoffs' in insights['trade_off_analysis']:
            print(f"  Strong Trade-offs: {insights['trade_off_analysis']['n_strong_tradeoffs']}")
    
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
    pipeline.add_step('step_phase1_model_training', step_phase1_model_training_and_evaluation, enabled=True)
    pipeline.add_step('step_phase1_decision_tree_rules', step_phase1_decision_tree_rules, enabled=True)
    pipeline.add_step('step_phase1_comprehensive_cluster_insights', step_phase1_comprehensive_cluster_insights, enabled=True)
    
    return pipeline


if __name__ == "__main__":
    main()
