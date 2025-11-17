"""
Visualize Rule Validation Results

Creates interactive plots to explore rule validation:
1. Precision Validation - Compare actual vs reported precision
2. Consistency Heatmap - Show consistency ratings across clusters/rules
3. Performance Comparison - Compare metrics across rules
4. Coverage vs Precision - Scatter plot showing rule quality
5. Metric Distribution - Box plots of key metrics per cluster
6. Recommendations Dashboard - Best rules for each metric
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys


def load_data(data_dir):
    """Load validation results and related data."""
    validation_df = pd.read_csv(data_dir / 'rule_validation_summary.csv')
    
    with open(data_dir / 'rule_validation_results.json', 'r') as f:
        validation_results = json.load(f)
    
    with open(data_dir / 'rule_recommendations.json', 'r') as f:
        recommendations = json.load(f)
    
    return validation_df, validation_results, recommendations


def plot_precision_validation(validation_df, output_dir):
    """Plot 1: Precision Validation - Actual vs Reported"""
    
    fig = go.Figure()
    
    # Perfect match line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Match',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    # Color by cluster
    for cluster_id in sorted(validation_df['cluster_id'].unique()):
        cluster_data = validation_df[validation_df['cluster_id'] == cluster_id]
        
        fig.add_trace(go.Scatter(
            x=cluster_data['reported_precision'],
            y=cluster_data['actual_precision'],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(
                size=cluster_data['n_matching'] / 2,  # Size by coverage
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Cluster {row['cluster_id']} - Rule {row['rule_number']}<br>"
                  f"Reported: {row['reported_precision']:.2%}<br>"
                  f"Actual: {row['actual_precision']:.2%}<br>"
                  f"Coverage: {row['n_matching']} workflows<br>"
                  f"Validated: {'✓' if row['precision_validated'] else '✗'}"
                  for _, row in cluster_data.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Rule Precision Validation: Actual vs Reported',
        xaxis_title='Reported Precision',
        yaxis_title='Actual Precision',
        hovermode='closest',
        width=900,
        height=700,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    
    fig.write_html(output_dir / 'plot1_precision_validation.html')
    print(f"✓ Created: plot1_precision_validation.html")
    
    return fig


def plot_consistency_heatmap(validation_df, output_dir):
    """Plot 2: Consistency Heatmap across Metrics"""
    
    # Define consistency order
    consistency_order = {'EXCELLENT': 4, 'GOOD': 3, 'MODERATE': 2, 'POOR': 1, 'N/A': 0}
    
    # Get key metrics
    metrics = ['Accuracy', 'Recall', 'Precision', 'AUC-ROC']
    
    # Create matrix
    consistency_matrix = []
    labels_y = []
    
    for _, row in validation_df.iterrows():
        rule_label = f"C{row['cluster_id']}-R{row['rule_number']}"
        labels_y.append(rule_label)
        
        row_values = []
        for metric in metrics:
            consistency_col = f"{metric}_consistency"
            if consistency_col in row:
                consistency = row[consistency_col]
                row_values.append(consistency_order.get(consistency, 0))
            else:
                row_values.append(0)
        
        consistency_matrix.append(row_values)
    
    consistency_matrix = np.array(consistency_matrix)
    
    # Create custom colorscale
    colorscale = [
        [0.0, '#d3d3d3'],    # N/A - gray
        [0.25, '#ff6b6b'],   # POOR - red
        [0.5, '#ffd93d'],    # MODERATE - yellow
        [0.75, '#6bcf7f'],   # GOOD - light green
        [1.0, '#4d96ff']     # EXCELLENT - blue
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=consistency_matrix,
        x=metrics,
        y=labels_y,
        colorscale=colorscale,
        text=[[f"{metrics[j]}<br>" + 
               list(consistency_order.keys())[list(consistency_order.values()).index(int(consistency_matrix[i, j]))]
               for j in range(len(metrics))]
              for i in range(len(labels_y))],
        hovertemplate='Rule: %{y}<br>Metric: %{x}<br>Consistency: %{text}<extra></extra>',
        colorbar=dict(
            title="Consistency",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['N/A', 'POOR', 'MODERATE', 'GOOD', 'EXCELLENT']
        )
    ))
    
    fig.update_layout(
        title='Rule Consistency Ratings by Metric',
        xaxis_title='Performance Metric',
        yaxis_title='Cluster-Rule',
        width=900,
        height=800,
        template='plotly_white'
    )
    
    fig.write_html(output_dir / 'plot2_consistency_heatmap.html')
    print(f"✓ Created: plot2_consistency_heatmap.html")
    
    return fig


def plot_performance_comparison(validation_df, output_dir):
    """Plot 3: Performance Comparison - Key Metrics Across Rules"""
    
    metrics = ['Accuracy_mean', 'Recall_mean', 'Precision_mean', 'AUC-ROC_mean']
    metric_labels = ['Accuracy', 'Recall', 'Precision', 'AUC-ROC']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        if metric in validation_df.columns:
            # Sort by cluster and metric
            sorted_df = validation_df.sort_values(['cluster_id', metric], ascending=[True, False])
            
            # Create bar chart
            for cluster_id in sorted(validation_df['cluster_id'].unique()):
                cluster_data = sorted_df[sorted_df['cluster_id'] == cluster_id]
                
                fig.add_trace(
                    go.Bar(
                        x=[f"C{row['cluster_id']}-R{row['rule_number']}" 
                           for _, row in cluster_data.iterrows()],
                        y=cluster_data[metric],
                        name=f'Cluster {cluster_id}',
                        text=cluster_data[metric].round(3),
                        textposition='auto',
                        showlegend=(idx == 0),
                        legendgroup=f'cluster_{cluster_id}',
                        hovertemplate=f'%{{x}}<br>{label}: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            # Update axes
            fig.update_yaxes(title_text=label, row=row, col=col)
            if row == 2:
                fig.update_xaxes(title_text='Rule', row=row, col=col, tickangle=-45)
    
    fig.update_layout(
        title='Performance Metrics Comparison Across Rules',
        height=800,
        width=1200,
        template='plotly_white',
        showlegend=True,
        barmode='group'
    )
    
    fig.write_html(output_dir / 'plot3_performance_comparison.html')
    print(f"✓ Created: plot3_performance_comparison.html")
    
    return fig


def plot_coverage_vs_precision(validation_df, output_dir):
    """Plot 4: Coverage vs Precision Scatter"""
    
    fig = go.Figure()
    
    for cluster_id in sorted(validation_df['cluster_id'].unique()):
        cluster_data = validation_df[validation_df['cluster_id'] == cluster_id]
        
        # Calculate quality score (precision * log(coverage))
        quality_scores = cluster_data['actual_precision'] * np.log1p(cluster_data['n_matching'])
        
        fig.add_trace(go.Scatter(
            x=cluster_data['n_matching'],
            y=cluster_data['actual_precision'],
            mode='markers+text',
            name=f'Cluster {cluster_id}',
            marker=dict(
                size=quality_scores * 10,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=[f"R{row['rule_number']}" for _, row in cluster_data.iterrows()],
            textposition='top center',
            hovertemplate='<b>Cluster %{fullData.name}</b><br>' +
                         'Coverage: %{x} workflows<br>' +
                         'Precision: %{y:.2%}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Rule Quality: Coverage vs Precision',
        xaxis_title='Coverage (Number of Matching Workflows)',
        yaxis_title='Actual Precision',
        hovermode='closest',
        width=1000,
        height=700,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.update_yaxes(range=[0, 1.05])
    
    # Add quality zones
    fig.add_shape(
        type="rect",
        x0=0, x1=validation_df['n_matching'].max(),
        y0=0.9, y1=1.05,
        fillcolor="lightgreen",
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    
    fig.add_annotation(
        x=validation_df['n_matching'].max() * 0.95,
        y=0.95,
        text="High Quality Zone",
        showarrow=False,
        font=dict(size=12, color="green"),
        opacity=0.5
    )
    
    fig.write_html(output_dir / 'plot4_coverage_vs_precision.html')
    print(f"✓ Created: plot4_coverage_vs_precision.html")
    
    return fig


def plot_metric_distributions(validation_results, output_dir):
    """Plot 5: Metric Distributions per Cluster"""
    
    key_metrics = ['Accuracy', 'Recall', 'Precision', 'AUC-ROC']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=key_metrics,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, metric in enumerate(key_metrics):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        # Collect data for each cluster
        for result in validation_results:
            cluster_id = result['cluster_id']
            
            if metric in result['performance_stats']:
                stats = result['performance_stats'][metric]
                
                # Create box plot data
                fig.add_trace(
                    go.Box(
                        name=f"C{cluster_id}-R{result['rule_number']}",
                        q1=[stats['q25']],
                        median=[stats['q50']],
                        q3=[stats['q75']],
                        lowerfence=[stats['min']],
                        upperfence=[stats['max']],
                        mean=[stats['mean']],
                        showlegend=False,
                        marker_color=px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)],
                        hovertemplate=f'<b>Cluster {cluster_id} - Rule {result["rule_number"]}</b><br>' +
                                     f'Mean: {stats["mean"]:.4f}<br>' +
                                     f'Std: {stats["std"]:.4f}<br>' +
                                     f'Consistency: {stats["consistency"]}<br>' +
                                     f'N: {stats["n_samples"]}<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        fig.update_yaxes(title_text=metric, row=row, col=col)
        if row == 2:
            fig.update_xaxes(title_text='Rule', row=row, col=col, tickangle=-45)
    
    fig.update_layout(
        title='Metric Distributions per Cluster-Rule',
        height=800,
        width=1400,
        template='plotly_white'
    )
    
    fig.write_html(output_dir / 'plot5_metric_distributions.html')
    print(f"✓ Created: plot5_metric_distributions.html")
    
    return fig


def plot_recommendations_dashboard(recommendations, validation_df, output_dir):
    """Plot 6: Recommendations Dashboard"""
    
    key_metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC-ROC']
    
    # Prepare data for table
    table_data = []
    
    for metric in key_metrics:
        if metric in recommendations:
            recs = recommendations[metric]
            
            # Highest mean
            if recs['highest_mean']:
                r = recs['highest_mean']
                table_data.append({
                    'Metric': metric,
                    'Goal': 'Highest Mean',
                    'Cluster': r['cluster'],
                    'Rule': r['rule_number'],
                    'Expected Value': f"{r['mean']:.4f} ± {r['std']:.4f}",
                    'Consistency': r['consistency'],
                    'Coverage': r['n_workflows']
                })
            
            # Most reliable
            if recs['most_reliable_high']:
                r = recs['most_reliable_high']
                table_data.append({
                    'Metric': metric,
                    'Goal': 'Most Reliable High',
                    'Cluster': r['cluster'],
                    'Rule': r['rule_number'],
                    'Expected Value': f"{r['mean']:.4f} ± {r['std']:.4f}",
                    'Consistency': r['consistency'],
                    'Coverage': r['n_workflows']
                })
    
    if not table_data:
        print("No recommendation data available")
        return None
    
    df_table = pd.DataFrame(table_data)
    
    # Create colored table
    fig = go.Figure(data=[go.Table(
        columnwidth=[80, 120, 60, 60, 120, 90, 70],
        header=dict(
            values=list(df_table.columns),
            fill_color='#4d96ff',
            font=dict(color='white', size=12),
            align='left',
            height=30
        ),
        cells=dict(
            values=[df_table[col] for col in df_table.columns],
            fill_color=[['#f0f0f0' if i % 2 == 0 else 'white' 
                        for i in range(len(df_table))]],
            align='left',
            height=25,
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title='Rule Recommendations for Target Metrics',
        height=max(400, len(table_data) * 30 + 100),
        width=1200,
        template='plotly_white'
    )
    
    fig.write_html(output_dir / 'plot6_recommendations_dashboard.html')
    print(f"✓ Created: plot6_recommendations_dashboard.html")
    
    # Also create a visual chart
    fig2 = go.Figure()
    
    for metric in key_metrics:
        if metric in recommendations and recommendations[metric]['highest_mean']:
            r = recommendations[metric]['highest_mean']
            
            fig2.add_trace(go.Bar(
                name=metric,
                x=[metric],
                y=[r['mean']],
                error_y=dict(type='data', array=[r['std']]),
                text=f"C{r['cluster']}-R{r['rule_number']}",
                textposition='auto',
                hovertemplate=f'<b>{metric}</b><br>' +
                             f'Mean: {r["mean"]:.4f}<br>' +
                             f'Std: {r["std"]:.4f}<br>' +
                             f'Consistency: {r["consistency"]}<br>' +
                             f'Cluster {r["cluster"]} - Rule {r["rule_number"]}<br>' +
                             f'Coverage: {r["n_workflows"]} workflows<extra></extra>'
            ))
    
    fig2.update_layout(
        title='Best Rules for Maximizing Each Metric',
        xaxis_title='Metric',
        yaxis_title='Expected Value (Mean ± Std)',
        height=600,
        width=1000,
        template='plotly_white',
        showlegend=False
    )
    
    fig2.write_html(output_dir / 'plot6b_best_rules_comparison.html')
    print(f"✓ Created: plot6b_best_rules_comparison.html")
    
    return fig


def create_index_html(output_dir):
    """Create an index.html to view all plots."""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rule Validation Visualizations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .plot-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            background: white;
        }
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .plot-card iframe {
            width: 100%;
            height: 500px;
            border: none;
        }
        .plot-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: 600;
        }
        .plot-description {
            padding: 15px 20px;
            color: #666;
            font-size: 0.95em;
            background: #f8f9fa;
        }
        .stats-box {
            background: #f0f7ff;
            border-left: 4px solid #4d96ff;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .full-width iframe {
            height: 700px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Rule Validation Visualizations</h1>
        <p class="subtitle">Interactive visualizations for decision rule validation results</p>
        
        <div class="stats-box">
            <strong>About These Visualizations:</strong><br>
            These plots help you understand the quality and reliability of extracted decision rules.
            Each visualization focuses on a different aspect of rule validation.
        </div>
        
        <div class="plot-grid">
            <div class="plot-card">
                <div class="plot-title">1. Precision Validation</div>
                <div class="plot-description">
                    Compare actual vs reported precision for each rule. Points on the diagonal line indicate perfect validation.
                    Marker size shows coverage (number of matching workflows).
                </div>
                <iframe src="plot1_precision_validation.html"></iframe>
            </div>
            
            <div class="plot-card">
                <div class="plot-title">2. Consistency Heatmap</div>
                <div class="plot-description">
                    Shows consistency ratings (EXCELLENT/GOOD/MODERATE/POOR) for each rule across different metrics.
                    Blue = Excellent, Green = Good, Yellow = Moderate, Red = Poor.
                </div>
                <iframe src="plot2_consistency_heatmap.html"></iframe>
            </div>
            
            <div class="plot-card full-width">
                <div class="plot-title">3. Performance Comparison</div>
                <div class="plot-description">
                    Compare mean performance metrics (Accuracy, Recall, Precision, AUC-ROC) across all rules.
                    Grouped by cluster for easy comparison.
                </div>
                <iframe src="plot3_performance_comparison.html"></iframe>
            </div>
            
            <div class="plot-card">
                <div class="plot-title">4. Coverage vs Precision</div>
                <div class="plot-description">
                    Scatter plot showing rule quality. High coverage + high precision = ideal rules.
                    Marker size indicates quality score (precision × log(coverage)).
                </div>
                <iframe src="plot4_coverage_vs_precision.html"></iframe>
            </div>
            
            <div class="plot-card">
                <div class="plot-title">5. Metric Distributions</div>
                <div class="plot-description">
                    Box plots showing the distribution of metrics for workflows matching each rule.
                    Narrow boxes indicate consistent performance.
                </div>
                <iframe src="plot5_metric_distributions.html"></iframe>
            </div>
            
            <div class="plot-card full-width">
                <div class="plot-title">6. Recommendations Dashboard</div>
                <div class="plot-description">
                    Best rules for maximizing each metric. Shows expected value with error bars (mean ± std).
                </div>
                <iframe src="plot6b_best_rules_comparison.html"></iframe>
            </div>
            
            <div class="plot-card full-width">
                <div class="plot-title">6b. Detailed Recommendations Table</div>
                <div class="plot-description">
                    Detailed table of recommendations showing cluster, rule number, expected performance, and consistency ratings.
                </div>
                <iframe src="plot6_recommendations_dashboard.html"></iframe>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created: index.html (main dashboard)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 5_visualize_rule_validation.py data/workflows")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    output_dir = data_dir / 'validation_plots'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("RULE VALIDATION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    validation_df, validation_results, recommendations = load_data(data_dir)
    print(f"✓ Loaded {len(validation_df)} rules")
    print()
    
    # Create plots
    print("Creating visualizations...")
    print()
    
    plot_precision_validation(validation_df, output_dir)
    plot_consistency_heatmap(validation_df, output_dir)
    plot_performance_comparison(validation_df, output_dir)
    plot_coverage_vs_precision(validation_df, output_dir)
    plot_metric_distributions(validation_results, output_dir)
    plot_recommendations_dashboard(recommendations, validation_df, output_dir)
    
    # Create index
    create_index_html(output_dir)
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\n🌐 Open {output_dir / 'index.html'} in your browser to view all plots")
    print()


if __name__ == '__main__':
    main()
