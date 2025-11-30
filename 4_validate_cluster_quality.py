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

# Publication-ready font size configuration
FONT_SIZE = {
    'title': 22,
    'xlabel': 20,
    'ylabel': 20,
    'xtick': 20,
    'ytick': 20,
    'legend': 20,
    'figure': 12,
    'figsize': [16, 6]
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

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.2, 1.1, 0.2))
    
    plt.suptitle(f"Silhouette analysis for clustering with n_clusters = {len(per_cluster_metrics_df['cluster_id'])}",
                 fontsize=14, fontweight='bold')
    
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
    
    ax.set_title('Distribution of Silhouette Scores per Cluster', fontsize=16)
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
    fig.suptitle('Overall Cluster Quality Scores', fontsize=16)
    
    colors = ['#4c72b0', '#55a868', '#c44e52']
    
    for i, (metric, value, color) in enumerate(zip(metrics, values, colors)):
        axes[i].bar(metric, value, color=color, edgecolor='white')
        axes[i].set_title(metric.replace('_', ' '), fontsize=12)
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
    plt.title('Predictive quality', fontsize=14)
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
    
    plt.title('Predictive Quality Metrics by Cluster', pad=20, fontweight='bold')
    
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
    ax.set_title('Stability of Discriminative Metrics within Rules\n(Bar = Median CV, Whiskers = 10th-90th Percentile Range)', fontsize=14)
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
    Creates boxplots showing the distribution of CV values for each cluster's rules.
    """
    # Get unique clusters and rules
    clusters = sorted(cv_summary_df['Cluster'].unique())
    
    # Setup plot
    fig, ax = plt.subplots()
    
    # Prepare data for boxplot: collect CV distributions for each cluster
    data_to_plot = []
    labels = []
    
    for cluster in clusters:
        cluster_data = cv_summary_df[cv_summary_df['Cluster'] == cluster]
        
        # For each cluster, create a list containing the CV percentile ranges
        # We'll use the median, 10th, and 90th percentiles to construct the distribution
        cv_values = []
        for _, row in cluster_data.iterrows():
            # Add the three key points: 10th percentile, median, 90th percentile
            cv_values.extend([row['CV_10th_Percentile'], row['Median_CV'], row['CV_90th_Percentile']])
        
        if cv_values:
            data_to_plot.append(cv_values)
            labels.append(f'Cluster {cluster}')
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels, widths=0.6)
    
    # Customize colors using a colormap
    colors = [plt.cm.viridis(float(i) / len(clusters)) for i in range(len(clusters))]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='--', alpha=0.7)
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    # ax.set_title('Stability of Discriminative Metrics within Rules\n(Distribution of CV values across rules per cluster)', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Rule quality plot saved to {save_path}")

if __name__ == "__main__":
    path="./data/workflows"
    dataset_name = "adult"
    result_dir_cluster_quality = f"./results/{dataset_name}/cluster_quality"
    result_dir_predictive_quality = f"./results/{dataset_name}/predictive_quality"
    result_dir_rule_quality = f"./results/{dataset_name}/rule_quality"


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
    predictive_df.to_csv(os.path.join(result_dir_predictive_quality, "predictive_quality_metrics.csv"), index=False)
    plot_predictive_quality(predictive_df, save_path=os.path.join(result_dir_predictive_quality, "predictive_quality.png"))
    plot_predictive_quality_table(predictive_df, save_path=os.path.join(result_dir_predictive_quality, "predictive_quality_table.png"))
       
    ##Rule quality
    rules_df = pd.read_csv("./data/workflows/cluster_decision_rules.csv")
    raw_data_full = pd.read_csv("./data/workflows/workflows.csv")
    cluster_labels = pd.read_csv("./data/workflows/workflows_clustered.csv")[['workflowId', 'cluster']]
    _,sub_frames=overall_for_rules(rules_df, raw_data_full, cluster_labels)
    cv_summary_df=rule_quality(sub_frames, clusters_insights)
    os.makedirs(result_dir_rule_quality, exist_ok=True)
    cv_summary_df.to_csv(os.path.join(result_dir_rule_quality, "rule_quality_metrics.csv"), index=False)
    plot_rule_quality(cv_summary_df, save_path=os.path.join(result_dir_rule_quality, "rule_quality.png"))
    plot_rule_quality_box_plot(cv_summary_df, save_path=os.path.join(result_dir_rule_quality, "rule_quality_boxplot.png"))






