"""
ML Workflow Clustering Script
==============================
This script clusters ML experiment runs based on their performance metrics,
identifies medoids for each cluster, and computes SHAP values to understand
feature importance for clustering decisions.

Alternative implementation without scikit-learn-extra dependency.
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from prince import MCA
from _correlation_utils import (
    compute_correlations_with_components,
    compute_eta_squared_with_components,
    detect_variable_types
)
import warnings
warnings.filterwarnings('ignore')


def apply_elbow_method_pca(X_scaled, max_components=None, variance_threshold=0.95):
    """
    Apply elbow method to determine optimal number of PCA components
    for numerical variables (SPCA simulation using standard PCA).
    
    Args:
        X_scaled: Scaled numerical data matrix
        max_components: Maximum number of components to test (default: min(n_features-1, n_samples-1))
        variance_threshold: Stop when cumulative variance exceeds this threshold
    
    Returns:
        optimal_n_components: Optimal number of components based on elbow
        pca_fit: Fitted PCA object
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance: Array of cumulative explained variance
    """
    if max_components is None:
        max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
    
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find optimal number of components based on variance threshold
    optimal_n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # If threshold not reached, use elbow method (largest decrease in variance)
    if cumulative_variance[-1] < variance_threshold:
        variance_diffs = np.diff(pca.explained_variance_ratio_)
        # Find elbow: steepest drop in variance explained
        elbow_point = np.argmax(np.diff(variance_diffs))
        optimal_n_components = max(elbow_point + 2, 2)  # At least 2 components
    
    print(f"\nPCA Analysis (Numerical Variables):")
    print(f"  Original dimensions: {X_scaled.shape[1]}")
    print(f"  Optimal components: {optimal_n_components}")
    print(f"  Cumulative variance explained: {cumulative_variance[optimal_n_components - 1]:.1%}")
    
    return optimal_n_components, pca, pca.explained_variance_ratio_, cumulative_variance


def apply_elbow_method_mca(X_cat, max_components=None, inertia_threshold=0.95):
    """
    Apply elbow method to determine optimal number of MCA components
    for categorical variables.
    
    Args:
        X_cat: Categorical data (DataFrame with categorical columns)
        max_components: Maximum number of components to test
        inertia_threshold: Stop when cumulative inertia exceeds this threshold
    
    Returns:
        optimal_n_components: Optimal number of components based on elbow
        mca_fit: Fitted MCA object
        inertia_ratio: Array of inertia ratios
        cumulative_inertia: Array of cumulative inertia explained
    """
    if max_components is None:
        max_components = min(X_cat.shape[0] - 1, X_cat.shape[1])
    
    mca = MCA(n_components=max_components, random_state=42)
    mca.fit(X_cat)
    
    # Get inertia and calculate cumulative inertia
    total_inertia = mca.total_inertia_
    inertia_ratio = mca.inertia_ / total_inertia if total_inertia > 0 else mca.inertia_
    cumulative_inertia = np.cumsum(inertia_ratio)
    
    # Find optimal number of components based on inertia threshold
    optimal_n_components = np.argmax(cumulative_inertia >= inertia_threshold) + 1
    
    # If threshold not reached, use elbow method
    if cumulative_inertia[-1] < inertia_threshold:
        inertia_diffs = np.diff(inertia_ratio)
        elbow_point = np.argmax(np.diff(inertia_diffs))
        optimal_n_components = max(elbow_point + 2, 1)  # At least 1 component
    
    print(f"\nMCA Analysis (Categorical Variables):")
    print(f"  Original dimensions: {X_cat.shape[1]}")
    print(f"  Optimal components: {optimal_n_components}")
    print(f"  Cumulative adjusted inertia explained: {cumulative_inertia[optimal_n_components - 1]:.1%}")
    
    return optimal_n_components, mca, inertia_ratio, cumulative_inertia


def reduce_dimensions(df, metric_cols, pca_variance_threshold, mca_inertia_threshold,
                      corr_threshold, eta_threshold):
    """
    Apply combined dimensionality reduction using SPCA for numerical variables
    and MCA for categorical variables with automatic component selection via elbow method.
    
    This function:
    1. Separates numerical and categorical variables
    2. Standardizes numerical variables and applies PCA (SPCA simulation)
    3. Applies MCA to categorical variables
    4. Uses elbow method to automatically determine optimal components
    5. Identifies relevant original variables based on correlation/eta-squared thresholds
    6. Names derived components based on most correlated original variables
    
    Args:
        df: Input DataFrame
        metric_cols: List of metric column names to include in dimensionality reduction
        pca_variance_threshold: Target cumulative variance for PCA
        mca_inertia_threshold: Target cumulative inertia for MCA
        corr_threshold: Correlation threshold for naming PCA components
        eta_threshold: Eta-squared threshold for naming MCA components
    
    Returns:
        reduced_df: DataFrame with derived components
        reduction_info: Dictionary with reduction details and component names
        component_correlations: Dictionary of correlations/eta-squared for each component
    """
    
    # Select only metric columns for reduction
    df_metrics = df[metric_cols].copy()
    
    # Identify numerical and categorical variables using unified approach
    var_types = detect_variable_types(df_metrics)
    numerical_cols = [col for col, vtype in var_types.items() if vtype == 'continuous']
    categorical_cols = [col for col, vtype in var_types.items() if vtype == 'categorical']
    
    print(f"\n{'='*60}")
    print("Dimensionality Reduction Analysis")
    print(f"{'='*60}")
    print(f"Original variables: {len(metric_cols)}")
    print(f"  - Numerical: {len(numerical_cols)}")
    print(f"  - Categorical: {len(categorical_cols)}")
    
    # Initialize dataframe for reduced components
    reduced_data = pd.DataFrame(index=df.index)
    component_names = []
    component_correlations = {}
    
    # ===== Process Numerical Variables with PCA =====
    if numerical_cols:
        X_numerical = df_metrics[numerical_cols].dropna(axis=0, how='all')
        
        # Apply low variance filter BEFORE standardization on original data
        filtered_numerical_cols, variance_filter_info = filter_low_variance_metrics(
            X_numerical, numerical_cols, cv_threshold=0.05
        )
        
        # Keep only non-filtered columns
        if len(filtered_numerical_cols) < len(numerical_cols):
            X_numerical = X_numerical[filtered_numerical_cols]
            numerical_cols = filtered_numerical_cols
        
        # Standardize numerical variables
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=X_numerical.index)
        
        # Apply elbow method for PCA with target variance threshold
        optimal_pca_components, pca_fit, explained_var, cumulative_var = apply_elbow_method_pca(
            X_numerical_scaled, 
            max_components=len(numerical_cols),
            variance_threshold=pca_variance_threshold
        )
        
        # Refit PCA with optimal number of components determined by elbow method
        pca_final = PCA(n_components=optimal_pca_components)
        pca_components = pca_final.fit_transform(X_numerical_scaled)
        
        # Create component names based on correlations
        pca_component_correlations = compute_correlations_with_components(
            X_numerical_scaled_df, pca_components, 
            [f'PC_{i+1}' for i in range(optimal_pca_components)],
            correlation_threshold=corr_threshold
        )

        # Generate meaningful names for PCA components
        for i in range(optimal_pca_components):
            relevant_vars = pca_component_correlations[f'PC_{i+1}']
            component_name = f"PC_{i+1}"
            # if relevant_vars:
                # Create a set of all relevant variable names
                # component_name = f"PC_{i+1}"
                # var_names = {var['variable'] for var in relevant_vars}
                # component_name = f"PCA_{i+1}_{{{','.join(sorted(var_names))}}}"
            # else:
            #     continue
            
            component_names.append(component_name)
            reduced_data[component_name] = pca_components[:, i]
            component_correlations[component_name] = relevant_vars
        
        print(f"\nPCA Components: {optimal_pca_components}")
        print(f"Variance explained: {pca_final.explained_variance_ratio_.sum():.1%}")
    
    # ===== Process Categorical Variables with MCA =====
    if categorical_cols:
        X_categorical = df_metrics[categorical_cols].copy()
        
        # Handle missing values in categorical data
        X_categorical = X_categorical.fillna('Unknown')
        # Apply elbow method for MCA with target inertia threshold
        optimal_mca_components, mca_fit, inertia, cumulative_inertia = apply_elbow_method_mca(
            X_categorical,
            max_components=len(categorical_cols),
            inertia_threshold=mca_inertia_threshold
        )
        
        # Refit MCA with optimal number of components determined by elbow method
        mca_final = MCA(n_components=optimal_mca_components, random_state=42)
        mca_components = mca_final.fit_transform(X_categorical).values
        
        # Create component names based on eta-squared
        mca_component_eta = compute_eta_squared_with_components(
            X_categorical, mca_components,
            [f'MCA_{i+1}' for i in range(optimal_mca_components)],
            eta_threshold=eta_threshold
        )
        
        # Generate meaningful names for MCA components
        for i in range(optimal_mca_components):
            relevant_cats = mca_component_eta[f'MCA_{i+1}']
            component_name = f'MCA_{i+1}'
            # if relevant_cats:
            #     # Create a set of all relevant categorical variable names
            #     cat_names = {cat['variable'] for cat in relevant_cats}
            #     component_name = f"MCA_{i+1}_{{{','.join(sorted(cat_names))}}}"
            # else:
            #     component_name = f'MCA_{i+1}'
            
            component_names.append(component_name)
            reduced_data[component_name] = mca_components[:, i]
            component_correlations[component_name] = relevant_cats
        
        print(f"\nMCA Components: {optimal_mca_components}")
        print(f"Inertia explained: {cumulative_inertia[optimal_mca_components - 1]:.1%}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Dimensionality Reduction Summary")
    print(f"{'='*60}")
    print(f"Original dimensions: {len(metric_cols)}")
    print(f"Derived dimensions: {len(component_names)}")
    print(f"Reduction rate: {(1 - len(component_names) / len(metric_cols)):.1%}")
    print(f"\nDerived components:")
    for comp_name in component_names:
        print(f"  - {comp_name}")
    
    # Create reduction info dictionary
    reduction_info = {
        'n_original_variables': len(metric_cols),
        'n_derived_variables': len(component_names),
        'n_numerical_original': len(numerical_cols),
        'n_categorical_original': len(categorical_cols),
        'n_pca_components': len([c for c in component_names if 'PCA' in c]),
        'n_mca_components': len([c for c in component_names if 'MCA' in c]),
        'component_names': component_names,
        'numerical_variables': numerical_cols,
        'categorical_variables': categorical_cols,
        'kept_metric_cols': numerical_cols + categorical_cols  # Track which metrics were actually kept after filtering
    }
    
    return reduced_data, reduction_info, component_correlations


def filter_low_variance_metrics(df, metric_cols, cv_threshold=0.1):
    """
    Filter out metrics with low variance using Coefficient of Variation (CV).
    
    CV = std / mean, which is scale-independent and works across metrics with different ranges.
    Metrics with CV below the threshold are considered uninformative for clustering.
    
    Args:
        df: Input DataFrame
        metric_cols: List of metric column names
        cv_threshold: Coefficient of Variation threshold. Metrics with CV below 
                     this are filtered out (default: 0.1 = 10%)
    
    Returns:
        filtered_metrics: List of metric column names after low variance filter
        filtered_info: Dictionary with filtering details
    """
    metric_variances = []
    
    for col in metric_cols:
        # Calculate variance statistics only for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            variance = df[col].var()
            
            # Coefficient of Variation: std / mean (scale-independent)
            # Handle edge cases: if mean is 0 or very close to 0, use variance as fallback
            if abs(mean_val) > 1e-10:
                cv = std_val / abs(mean_val)
            else:
                cv = variance  # Fallback for metrics with mean near 0
            
            metric_variances.append({
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'variance': variance,
                'cv': cv,
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min()
            })
        else:
            # Non-numeric columns are kept (will be handled as categorical)
            metric_variances.append({
                'metric': col,
                'mean': np.nan,
                'std': np.nan,
                'variance': np.inf,
                'cv': np.inf,  # Non-numeric columns are always kept
                'min': np.nan,
                'max': np.nan,
                'range': np.nan
            })
    
    variance_df = pd.DataFrame(metric_variances)
    
    # Identify low variance metrics based on CV threshold
    low_variance_metrics = variance_df[(variance_df['cv'] < cv_threshold) & (variance_df['cv'] != np.inf)]['metric'].tolist()
    filtered_metrics = [col for col in metric_cols if col not in low_variance_metrics]
    
    # Print filtering results
    print(f"\n{'='*80}")
    print("Low Variance Filter Analysis (Coefficient of Variation)")
    print(f"{'='*80}")
    print(f"Threshold (CV): {cv_threshold:.3f} (metrics with CV < {cv_threshold:.3f} are removed)")
    print(f"\nOriginal metrics: {len(metric_cols)}")
    print(f"Metrics removed (low CV): {len(low_variance_metrics)}")
    print(f"Metrics retained: {len(filtered_metrics)}")
    
    if low_variance_metrics:
        print(f"\nRemoved metrics (low coefficient of variation):")
        for metric in low_variance_metrics:
            var_info = variance_df[variance_df['metric'] == metric].iloc[0]
            print(f"  - {metric:30s} (CV: {var_info['cv']:.4f}, mean: {var_info['mean']:10.4f}, std: {var_info['std']:10.6f}, range: [{var_info['min']:.4f}, {var_info['max']:.4f}])")
    
    print(f"\nRetained metrics:")
    for metric in filtered_metrics:
        var_info = variance_df[variance_df['metric'] == metric].iloc[0]
        if pd.api.types.is_numeric_dtype(df[metric]):
            print(f"  - {metric:30s} (CV: {var_info['cv']:.4f}, mean: {var_info['mean']:10.4f}, std: {var_info['std']:10.6f}, range: [{var_info['min']:.4f}, {var_info['max']:.4f}])")
        else:
            print(f"  - {metric:30s} (categorical)")
    
    filtered_info = {
        'total_original': len(metric_cols),
        'total_removed': len(low_variance_metrics),
        'total_retained': len(filtered_metrics),
        'removed_metrics': low_variance_metrics,
        'cv_threshold': cv_threshold,
        'variance_details': variance_df.to_dict('records')
    }
    
    return filtered_metrics, filtered_info


def load_and_preprocess_data(data_folder='data', params_file=None, metrics_file=None):
    """Load workflow data and separate hyperparameters from metrics."""
    
    # If params_file and metrics_file not provided, use defaults from data_folder
    if params_file is None:
        params_file = os.path.join(data_folder, 'parameter_names.txt')
    if metrics_file is None:
        metrics_file = os.path.join(data_folder, 'metric_names.txt')
    
    # Load workflows CSV from folder
    filepath = os.path.join(data_folder, 'workflows.csv')
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Load hyperparameters from file if provided
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            hyperparams = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Define hyperparameters and metrics (fallback)
        hyperparams = ['criterion', 'fairness method', 'random state', 
                       'max depth', 'normalization', 'n estimators']

    # Load metrics from file if provided
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Exclude system metrics and identifiers
        exclude_cols = ['workflowId'] + hyperparams 
        metrics = [col for col in df.columns if col not in exclude_cols]

    print(f"Loaded {len(df)} workflows")
    print(f"Hyperparameters: {len(hyperparams)}")
    print(f"Performance metrics: {len(metrics)}")

    return df, hyperparams, metrics


def encode_hyperparameters(df, hyperparam_cols):
    """Encode categorical hyperparameters for clustering."""
    df_encoded = df.copy()
    label_encoders = {}

    for col in hyperparam_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df_encoded, label_encoders


def cluster_workflows(X_scaled, n_clusters=4, random_state=42):
    """
    Cluster workflows based on performance metrics using KMeans,
    then find medoids (actual data points closest to centroids).
    """
    # Perform KMeans clustering
    print(f"\nClustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Find medoids (actual workflows closest to centroids)
    medoid_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)

    print(f"Clustering complete. Silhouette score: {compute_silhouette(X_scaled, cluster_labels):.3f}")

    return cluster_labels, medoid_indices


def compute_silhouette(X, labels):
    """Compute silhouette score for clustering quality."""
    try:
        return silhouette_score(X, labels)
    except:
        return 0.0


def identify_small_clusters(cluster_labels, n_std=1.5):
    """
    Statistically identify small clusters based on size distribution.
    
    A cluster is considered "small" if its size is more than n_std standard 
    deviations below the mean cluster size.
    
    Args:
        cluster_labels: Array of cluster assignments
        n_std: Number of standard deviations below mean to define "small"
    
    Returns:
        small_cluster_ids: Set of cluster IDs that are statistically small
        cluster_sizes: Dict of cluster_id -> size
        stats_info: Dict with mean, std, threshold for reporting
    """
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, cluster_counts))
    
    sizes = np.array(cluster_counts)
    mean_size = sizes.mean()
    std_size = sizes.std()
    
    # Small cluster threshold: below mean - n_std * std
    small_threshold = max(mean_size - n_std * std_size, 2) # Ensure at least 2
    small_cluster_ids = set()
    for cluster_id, size in cluster_sizes.items():
        if size < small_threshold:
            small_cluster_ids.add(cluster_id)
    
    stats_info = {
        'mean_size': mean_size,
        'std_size': std_size,
        'small_threshold': small_threshold,
        'n_std': n_std
    }
    
    return small_cluster_ids, cluster_sizes, stats_info


def save_clustering_results(df, cluster_labels, medoid_indices, X, column_names, output_folder='data', output_prefix='workflows', kept_cols=None):
    """Save clustering results to files in the output folder, including scaled dataset.
    
    Args:
        df: Original dataframe with all columns
        cluster_labels: Cluster assignments for each sample
        medoid_indices: Indices of medoid samples
        X: Scaled data used for clustering
        column_names: Names of columns in X (after dimensionality reduction)
        output_folder: Output directory
        output_prefix: Prefix for output filenames
        kept_metrics: List of original metric columns that were kept after low-variance filtering.
                     If provided, only these metrics are saved in the clustered output.
    """

    # Add cluster labels to original data
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Filter to only kept metrics if provided
    if kept_cols is not None:
        # Keep workflowId and kept metrics, then add cluster label
        cols_to_keep = ['workflowId'] + kept_cols + ['cluster']
        # Only include columns that exist in the dataframe
        cols_to_keep = [col for col in cols_to_keep if col in df_clustered.columns]
        df_clustered = df_clustered[cols_to_keep]

    # Save clustered workflows
    output_file = os.path.join(output_folder, f'{output_prefix}_clustered.csv')
    df_clustered.to_csv(output_file, index=False)
    print(f"\nSaved clustered workflows to: {output_file}")

    # Save processed dataset for reuse in stats generation
    processed_df = pd.DataFrame(X, columns=column_names)
    processed_df['cluster'] = cluster_labels
    processed_file = os.path.join(output_folder, f'{output_prefix}_processed_data.csv')
    processed_df.to_csv(processed_file, index=False)
    print(f"Saved processed dataset to: {processed_file}")

    # Save medoid information
    medoid_data = []
    for cluster_id, medoid_idx in enumerate(medoid_indices):
        medoid_data.append({
            'cluster_id': cluster_id,
            'medoid_index': medoid_idx,
            'workflow_id': df.iloc[medoid_idx]['workflowId']
        })

    medoid_df = pd.DataFrame(medoid_data)
    medoid_file = os.path.join(output_folder, f'{output_prefix}_medoids.csv')
    medoid_df.to_csv(medoid_file, index=False)
    print(f"Saved medoid information to: {medoid_file}")

    return df_clustered


class ClusteringPipeline:
    """
    Modular pipeline for ML workflow clustering.
    Allows enabling/disabling and reordering steps for testing.
    Steps can be optional and dependent steps handle missing results gracefully.
    """
    
    def __init__(self):
        """Initialize pipeline state."""
        self.steps = []
        self.results = {}
        self.enabled_steps = set()
        self.skip_missing_deps = True  # Enable graceful handling of missing dependencies
    
    def add_step(self, name, func, enabled=True):
        """
        Add a step to the pipeline.
        
        Args:
            name: Unique identifier for the step
            func: Callable that performs the step logic
            enabled: Whether this step is enabled by default
        """
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
        """
        Safely retrieve results from a previous step.
        
        Args:
            step_name: Name of the step to get results from
            key: Optional key within the step's result dict
            default: Default value if step or key not found
        
        Returns:
            Result value or default if not found
        """
        if step_name not in self.results:
            return default
        
        result = self.results[step_name]
        if key is None:
            return result
        
        if isinstance(result, dict):
            return result.get(key, default)
        
        return default
    
    def run(self, **kwargs):
        """
        Execute the pipeline, running only enabled steps in order.
        
        Args:
            **kwargs: Arguments passed to each step function
        
        Returns:
            Dictionary of results from all executed steps
        """
        print("\n" + "="*60)
        print("ML Workflow Clustering Pipeline")
        print("="*60)
        
        for step in self.steps:
            step_name = step['name']
            
            if step_name not in self.enabled_steps:
                print(f"\n⊘ Skipping: {step_name}")
                continue
            
            print(f"\n▶ Running: {step_name}")
            print("-" * 60)
            
            try:
                # Pass both general kwargs and previously stored results
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


# ============ PIPELINE STEPS ============

def step_load_data(results, pipeline, **kwargs):
    """Step 1: Load and preprocess data."""
    data_folder = kwargs.get('data_folder', 'data')
    
    df, hyperparam_cols, metric_cols = load_and_preprocess_data(data_folder)
    
    return {
        'df': df,
        'hyperparam_cols': hyperparam_cols,
        'metric_cols': metric_cols,
        'data_folder': data_folder
    }


def step_dimensionality_reduction(results, pipeline, **kwargs):
    """Step 2: Apply dimensionality reduction."""
    # Get data from previous step (required)
    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')
    metric_cols = load_result.get('metric_cols')

    if df is None or metric_cols is None:
        raise KeyError("step_load_data: 'df' or 'metric_cols' not available. Load data first.")
    
    pca_variance_threshold = kwargs.get('pca_variance_threshold', 0.95)
    mca_inertia_threshold = kwargs.get('mca_inertia_threshold', 0.95)
    corr_threshold = kwargs.get('corr_threshold', 0.5)
    eta_threshold = kwargs.get('eta_threshold', 0.5)
    
    print("\n" + "="*60)
    print("Step 2: Dimensionality Reduction")
    print("="*60)
    
    reduced_data, reduction_info, component_correlations = reduce_dimensions(
        df, 
        metric_cols=metric_cols,
        pca_variance_threshold=pca_variance_threshold,  
        mca_inertia_threshold=mca_inertia_threshold,   
        corr_threshold=corr_threshold,
        eta_threshold=eta_threshold
    )
    
    return {
        'reduced_data': reduced_data,
        'reduction_info': reduction_info,
        'component_correlations': component_correlations
    }


def step_save_correlations(results, pipeline, **kwargs):
    """Step 2.5: Save component correlation information. (Optional - skipped if dim reduction is skipped)"""
    # Gracefully handle missing dimensionality reduction
    dim_result = pipeline.get_result('step_dimensionality_reduction', default=None)
    if dim_result is None:
        print("Skipping correlation save: dimensionality reduction not performed.")
        return {'correlation_summary': []}
    
    component_correlations = dim_result.get('component_correlations', {})
    
    # Get data_folder from load_data step
    load_result = pipeline.get_result('step_load_data', default={})
    data_folder = load_result.get('data_folder', 'data')
    
    correlation_summary = []
    for comp_name, correlations in component_correlations.items():
        for corr_info in correlations:
            if 'correlation' in corr_info:
                correlation_summary.append({
                    'component': comp_name,
                    'variable': corr_info['variable'],
                    'correlation': corr_info['correlation']
                })
            elif 'eta_squared' in corr_info:
                correlation_summary.append({
                    'component': comp_name,
                    'variable': corr_info['variable'],
                    'eta_squared': corr_info['eta_squared']
                })
    
    if correlation_summary:
        corr_df = pd.DataFrame(correlation_summary)
        corr_file = os.path.join(data_folder, 'workflows_component_correlations.csv')
        corr_df.to_csv(corr_file, index=False)
        print(f"Saved component correlations to: {corr_file}")
    
    return {'correlation_summary': correlation_summary}


def step_prepare_clustering_data(results, pipeline, **kwargs):
    """Step 3: Prepare and scale data for clustering."""
    # Try to get reduced data from dimensionality reduction
    dim_result = pipeline.get_result('step_dimensionality_reduction', default=None)
    
    if dim_result is None:
        # Fallback: use raw data from load step
        print("Dimensionality reduction not performed. Using raw metric data.")
        load_result = pipeline.get_result('step_load_data', default={})
        df = load_result.get('df')
        metric_cols = load_result.get('metric_cols')
        
        if df is None or metric_cols is None:
            raise KeyError("step_load_data: 'df' or 'metric_cols' not available. Load data first.")
        
        reduced_data = df[metric_cols].copy()
    else:
        reduced_data = dim_result.get('reduced_data')
    
    print("\n" + "="*60)
    print("Step 3: Prepare Data for Clustering")
    print("="*60)
    
    X_scaled = reduced_data.values
    
    return {'X_scaled': X_scaled, 'reduced_data': reduced_data}


def step_find_optimal_clusters(results, pipeline, **kwargs):
    """Step 4: Find optimal number of clusters using silhouette score."""
    cluster_result = pipeline.get_result('step_prepare_clustering_data', default=None)
    if cluster_result is None:
        raise KeyError("step_prepare_clustering_data: Data preparation required before finding optimal clusters.")
    
    X_scaled = cluster_result.get('X_scaled')
    reduced_data = cluster_result.get('reduced_data')
    
    min_k = kwargs.get('min_k', 3)
    max_k = kwargs.get('max_k', 9)
    
    print("\nFinding optimal number of clusters...")
    silhouette_scores = []
    
    for k in range(min_k, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced_data)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}, Silhouette Score: {score:.3f}")
    
    optimal_k = list(range(min_k, max_k))[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    return {'optimal_k': optimal_k, 'silhouette_scores': silhouette_scores}


def step_perform_clustering(results, pipeline, **kwargs):
    """Step 5: Perform clustering with optimal k or use provided k."""
    cluster_result = pipeline.get_result('step_prepare_clustering_data', default=None)
    if cluster_result is None:
        raise KeyError("step_prepare_clustering_data: Data preparation required before clustering.")
    
    X_scaled = cluster_result.get('X_scaled')
    
    # Try to get optimal_k from previous step, otherwise use kwargs or default
    optimal_k = kwargs.get('n_clusters', None)
    if optimal_k is None:
        opt_result = pipeline.get_result('step_find_optimal_clusters', default=None)
        if opt_result is not None:
            optimal_k = opt_result.get('optimal_k')
        else:
            optimal_k = kwargs.get('n_clusters', 4)  # Default to 4 if not found
    
    cluster_labels, medoid_indices = cluster_workflows(
        X_scaled, n_clusters=optimal_k
    )
    
    return {
        'cluster_labels': cluster_labels,
        'medoid_indices': medoid_indices,
        'n_clusters': optimal_k
    }


def step_identify_small_clusters(results, pipeline, **kwargs):
    """Step 6: Identify small clusters statistically. (Optional)"""
    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping small cluster identification: clustering not performed.")
        return {
            'small_clusters': set(),
            'cluster_sizes': {},
            'stats_info': {}
        }
    
    cluster_labels = cluster_result.get('cluster_labels')
    n_std = kwargs.get('n_std', 1.5)
    
    small_clusters, cluster_sizes, stats_info = identify_small_clusters(
        cluster_labels, n_std=n_std
    )
    
    return {
        'small_clusters': small_clusters,
        'cluster_sizes': cluster_sizes,
        'stats_info': stats_info
    }


def step_create_cluster_metadata(results, pipeline, **kwargs):
    """Step 7: Create cluster metadata with outlier flags. (Optional)"""
    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping cluster metadata: clustering not performed.")
        return {'cluster_metadata_df': pd.DataFrame()}
    
    cluster_labels = cluster_result.get('cluster_labels')
    optimal_k = cluster_result.get('n_clusters')
    
    small_result = pipeline.get_result('step_identify_small_clusters', default={})
    small_clusters = small_result.get('small_clusters', set())
    cluster_sizes = small_result.get('cluster_sizes', {})
    
    # Get data_folder from load_data step
    load_result = pipeline.get_result('step_load_data', default={})
    data_folder = load_result.get('data_folder', 'data')
    
    cluster_metadata = []
    for cluster_id in range(optimal_k):
        cluster_metadata.append({
            'cluster_id': cluster_id,
            'size': cluster_sizes.get(cluster_id, 0),
            'is_small': cluster_id in small_clusters,
            'outlier_status': 'SMALL' if cluster_id in small_clusters else 'NORMAL'
        })
    
    cluster_metadata_df = pd.DataFrame(cluster_metadata)
    cluster_metadata_file = os.path.join(data_folder, 'workflows_cluster_metadata.csv')
    cluster_metadata_df.to_csv(cluster_metadata_file, index=False)
    print(f"\nSaved cluster metadata to: {cluster_metadata_file}")
    
    return {'cluster_metadata_df': cluster_metadata_df}


def step_save_results(results, pipeline, **kwargs):
    """Step 8: Save all clustering results."""
    # Required dependencies
    load_result = pipeline.get_result('step_load_data', default={})
    df = load_result.get('df')
    data_folder = load_result.get('data_folder', 'data')
    hyperparam_cols = load_result.get('hyperparam_cols', [])

    cluster_result = pipeline.get_result('step_perform_clustering', default=None)
    if cluster_result is None:
        print("Skipping save: clustering not performed.")
        return {'df_clustered': None}
    
    cluster_labels = cluster_result.get('cluster_labels')
    medoid_indices = cluster_result.get('medoid_indices')
    
    prepare_result = pipeline.get_result('step_prepare_clustering_data', default={})
    X_scaled = prepare_result.get('X_scaled')
    
    # Get component names from dimensionality reduction if available
    dim_result = pipeline.get_result('step_dimensionality_reduction', default={})
    reduction_info = dim_result.get('reduction_info', {})
    component_names = reduction_info.get('component_names', [])
    kept_cols = hyperparam_cols + reduction_info.get('kept_metric_cols', None)
    
    # Fallback to metric columns if no component names
    if not component_names:
        component_names = load_result.get('metric_cols', [])
    
    df_clustered = save_clustering_results(
        df, cluster_labels, medoid_indices, X_scaled, component_names,
        output_folder=data_folder, kept_cols=kept_cols
    )
    
    print("\n" + "="*60)
    print("Clustering complete! Files generated in folder:")
    print(f"  {data_folder}/")
    print("  - workflows_clustered.csv")
    print("  - workflows_processed_data.csv")
    print("  - workflows_cluster_metadata.csv (with small cluster flags)")
    print("  - workflows_medoids.csv")
    print("="*60)
    
    return {'df_clustered': df_clustered}


def build_default_pipeline():
    """Build the default clustering pipeline with all steps."""
    pipeline = ClusteringPipeline()
    
    # Add all steps in order
    pipeline.add_step('step_load_data', step_load_data, enabled=True)
    pipeline.add_step('step_dimensionality_reduction', step_dimensionality_reduction, enabled=True)
    pipeline.add_step('step_save_correlations', step_save_correlations, enabled=True)
    pipeline.add_step('step_prepare_clustering_data', step_prepare_clustering_data, enabled=True)
    pipeline.add_step('step_find_optimal_clusters', step_find_optimal_clusters, enabled=True)
    pipeline.add_step('step_perform_clustering', step_perform_clustering, enabled=True)
    pipeline.add_step('step_identify_small_clusters', step_identify_small_clusters, enabled=True)
    pipeline.add_step('step_create_cluster_metadata', step_create_cluster_metadata, enabled=True)
    pipeline.add_step('step_save_results', step_save_results, enabled=True)
    
    return pipeline


def main():
    """Main execution function with modular pipeline."""
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python 1_cluster_workflows.py <data_folder>")
        print("\nArguments:")
        print("  data_folder - Path to folder containing workflows.csv, parameter_names.txt, and metric_names.txt")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    
    # Build the pipeline
    pipeline = build_default_pipeline()
    
    # Customize which steps to run (examples):
    # ==========================================
    
    # Example 1: Skip dimensionality reduction (use raw metrics)
    # pipeline.disable_step('step_dimensionality_reduction')
    # pipeline.disable_step('step_save_correlations')
    
    # Example 2: Skip optional analysis steps
    # pipeline.disable_step('step_identify_small_clusters')
    # pipeline.disable_step('step_create_cluster_metadata')
    
    # Example 3: Run only certain steps
    # pipeline.set_steps([
    #     'step_load_data', 
    #     'step_dimensionality_reduction',
    #     'step_prepare_clustering_data'
    # ])
    
    # Example 4: Skip k optimization (use fixed k value)
    # pipeline.disable_step('step_find_optimal_clusters')
    
    # Print current pipeline configuration
    pipeline.list_steps()
    
    # Configure pipeline parameters
    pipeline_params = {
        'data_folder': data_folder,
        'pca_variance_threshold': 0.8,
        'mca_inertia_threshold': 0.8,
        'corr_threshold': 0.75,
        'eta_threshold': 0.33,
        'min_k': 2,
        'max_k': 20,
        'n_std': 1.5,
        # 'n_clusters': 2  # Used if step_find_optimal_clusters is disabled
    }
    
    # Run the pipeline
    results = pipeline.run(**pipeline_params)
    
    return results.get('step_save_results', {}).get('df_clustered'), \
           results.get('step_perform_clustering', {}).get('cluster_labels')


if __name__ == "__main__":
    main()
