#!/usr/bin/env python3
"""
Simple Constrained Parallel Coordinates Visualization
YOU choose hyperparams and metrics - no auto-detection!
"""

import pandas as pd
import argparse
import webbrowser
import os
import time
from parallel_coordinates_constrained import (
    ConstrainedParallelCoordinatesOptimizer,
    create_parallel_coordinates
)


def select_columns_simple(data: pd.DataFrame) -> dict:
    """Simple column selection - just numbers, no guessing."""
    print("\n" + "="*70)
    print("🎯 CONSTRAINED PARALLEL COORDINATES")
    print("   You select: HYPERPARAMETERS → METRICS")
    print("="*70)
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Get categorical columns (string/object types)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categorical columns as numbers
    data_encoded = data.copy()
    categorical_mapping = {}
    
    for col in categorical_cols:
        # Convert to numeric codes (0, 1, 2, ...)
        data_encoded[col] = pd.Categorical(data[col]).codes
        unique_values = data[col].unique()
        categorical_mapping[col] = {i: val for i, val in enumerate(unique_values)}
    
    # All columns we can use (numeric + encoded categorical)
    all_usable_cols = numeric_cols + categorical_cols
    
    print(f"\n📊 Dataset: {len(data)} records, {len(data.columns)} columns")
    print(f"   Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
    print(f"   Total usable: {len(all_usable_cols)}")
    
    print("\n📋 Available columns:")
    print("-"*70)
    for i, col in enumerate(all_usable_cols, 1):
        col_type = "📊 numeric" if col in numeric_cols else "🏷️  categorical (encoded)"
        print(f"   {i:2d}. {col:40s} [{col_type}]")
    print("-"*70)
    
    if categorical_cols:
        print("\n💡 Categorical columns will be encoded as numbers:")
        for col in categorical_cols[:3]:  # Show first 3 as examples
            values = list(categorical_mapping[col].values())[:3]
            print(f"   • {col}: {', '.join(map(str, values))}{'...' if len(categorical_mapping[col]) > 3 else ''}")
        if len(categorical_cols) > 3:
            print(f"   ... and {len(categorical_cols) - 3} more")
        print()
    
    # Get hyperparameters
    print("\n⚙️  STEP 1: Select HYPERPARAMETERS (inputs)")
    print("   Enter column numbers separated by commas")
    print("   Example: 1,3,5,7")
    hp_input = input("\n   Hyperparam numbers: ").strip()
    hp_indices = [int(x.strip())-1 for x in hp_input.split(',') if x.strip()]
    selected_hyperparams = [all_usable_cols[i] for i in hp_indices]
    
    # Get metrics
    print("\n📈 STEP 2: Select METRICS (outputs)")
    print("   Enter column numbers separated by commas")
    print("   Example: 2,4,6,8,9,10")
    metric_input = input("\n   Metric numbers: ").strip()
    metric_indices = [int(x.strip())-1 for x in metric_input.split(',') if x.strip()]
    selected_metrics = [all_usable_cols[i] for i in metric_indices]
    
    # Show what was selected
    print("\n" + "="*70)
    print("✅ YOUR SELECTION:")
    print("="*70)
    print(f"\n⚙️  HYPERPARAMETERS ({len(selected_hyperparams)}):")
    for hp in selected_hyperparams:
        print(f"   • {hp}")
    
    print(f"\n📈 METRICS ({len(selected_metrics)}):")
    for m in selected_metrics:
        print(f"   • {m}")
    
    # Choose color metric
    print("\n🎨 STEP 3: Select COLOR METRIC")
    print("   Which metric should color the lines?")
    for i, col in enumerate(selected_metrics, 1):
        print(f"   {i}. {col}")
    
    color_idx = input(f"\n   Enter number (default=1): ").strip()
    color_col = selected_metrics[int(color_idx)-1] if color_idx and int(color_idx) <= len(selected_metrics) else selected_metrics[0]
    
    print(f"\n   Color: {color_col}")
    print("="*70)
    
    return {
        'hyperparams': selected_hyperparams,
        'metrics': selected_metrics,
        'color': color_col,
        'data_encoded': data_encoded,
        'categorical_mapping': categorical_mapping
    }


def create_constrained_summary(output_path: str, stats: dict):
    """Create summary highlighting the constrained optimization."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Constrained Parallel Coordinates Results</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-top: 0;
        }}
        .constraint-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
            font-size: 14px;
        }}
        .flow-diagram {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 30px;
            border-radius: 12px;
            margin: 30px 0;
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            color: #0d47a1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .flow-arrow {{
            font-size: 36px;
            color: #f5576c;
            margin: 0 20px;
        }}
        .header-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }}
        .comparison-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        .comparison-card h4 {{
            margin-top: 0;
            color: #667eea;
        }}
        .comparison-card.constrained {{
            border: 3px solid #28a745;
            background: #d4edda;
        }}
        .comparison-card.unconstrained {{
            border: 3px solid #6c757d;
            background: #e2e3e5;
        }}
        .metric {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric.constrained {{
            color: #28a745;
        }}
        .metric.unconstrained {{
            color: #6c757d;
        }}
        .improvement {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #0d47a1;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
            font-size: 20px;
            text-align: center;
            font-weight: 600;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        .viz-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        .viz-card h3 {{
            color: #333;
            margin-top: 0;
        }}
        .viz-card a {{
            display: inline-block;
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin-top: 15px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        .viz-card a:hover {{
            transform: scale(1.05);
        }}
        .crossings {{
            font-size: 28px;
            font-weight: bold;
            margin: 15px 0;
        }}
        .best {{ color: #28a745; }}
        .middle {{ color: #ffc107; }}
        .dimension-list {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .dimension-list h4 {{
            margin-top: 0;
        }}
        .dimension-section {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 20px;
            border-radius: 6px;
        }}
        .hyperparams-section {{
            background: #d1ecf1;
            border-left: 4px solid #0c5460;
        }}
        .metrics-section {{
            background: #d4edda;
            border-left: 4px solid #155724;
        }}
        .dimension-section strong {{
            display: block;
            margin-bottom: 5px;
        }}
        .boundary-info {{
            background: #ffe5e5;
            border: 2px dashed #ff0000;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            text-align: center;
        }}
        .boundary-info strong {{
            color: #dc3545;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            🎯 Constrained Parallel Coordinates
            <span class="constraint-badge">HYPERPARAMS → METRICS</span>
        </h1>
        
        <div class="flow-diagram">
            ⚙️ INPUTS
            <span class="flow-arrow">→</span>
            🔄 PROCESS
            <span class="flow-arrow">→</span>
            📈 OUTPUTS
            <br>
            <small style="font-size: 16px; opacity: 0.8;">
                Hyperparameters always before Metrics (logical flow preserved)
            </small>
        </div>
        
        <div class="header-stats">
            <div class="stat-card">
                <div class="label">Total Records</div>
                <div class="value">{stats['n_records']:,}</div>
            </div>
            <div class="stat-card">
                <div class="label">Hyperparameters</div>
                <div class="value">{len(stats['hyperparams'])}</div>
            </div>
            <div class="stat-card">
                <div class="label">Metrics</div>
                <div class="value">{len(stats['metrics'])}</div>
            </div>
            <div class="stat-card">
                <div class="label">Computation Time</div>
                <div class="value">{stats['time']:.2f}s</div>
            </div>
        </div>
        
        <div class="comparison-grid">
            <div class="comparison-card constrained">
                <h4>✅ Constrained (This Approach)</h4>
                <p><strong>Order:</strong> All hyperparams → All metrics</p>
                <div class="metric constrained">{stats['constrained_crosses']:,}</div>
                <p style="font-size: 12px; color: #666;">crossings</p>
                <ul style="text-align: left;">
                    <li>Preserves logical flow</li>
                    <li>Easy to interpret</li>
                    <li>Clear input→output relationship</li>
                </ul>
            </div>
            
            <div class="comparison-card unconstrained">
                <h4>❓ Unconstrained (Alternative)</h4>
                <p><strong>Order:</strong> Mixed hyperparams & metrics</p>
                <div class="metric unconstrained">{stats['unconstrained_crosses']:,}</div>
                <p style="font-size: 12px; color: #666;">crossings</p>
                <ul style="text-align: left;">
                    <li>Slightly fewer crossings</li>
                    <li>Harder to interpret</li>
                    <li>Breaks logical grouping</li>
                </ul>
            </div>
        </div>
        
        <div class="improvement">
            🎯 <strong>Result:</strong> {stats['improvement']:.1f}% reduction in edge crossings!
            <br>From {stats['original']:,} → {stats['constrained_crosses']:,} crossings
            <br>
            <small>(Trade-off: {stats['unconstrained_crosses']:,} possible with unconstrained, 
            but loses interpretability)</small>
        </div>
        
        <h2>📊 Visualizations</h2>
        <div class="viz-grid">
            <div class="viz-card">
                <h3>✅ Constrained Optimal</h3>
                <p>Best ordering with hyperparam→metric constraint</p>
                <div class="crossings best">{stats['constrained_crosses']:,}</div>
                <div style="color: #666; font-size: 12px;">crossings</div>
                <a href="constrained_optimal.html" target="_blank">View Visualization →</a>
            </div>
            
            <div class="viz-card">
                <h3>📊 Original Order</h3>
                <p>Dataset's natural ordering (baseline)</p>
                <div class="crossings middle">{stats['original']:,}</div>
                <div style="color: #666; font-size: 12px;">crossings</div>
                <a href="constrained_original.html" target="_blank">View Visualization →</a>
            </div>
            
            <div class="viz-card">
                <h3>🔓 Unconstrained Optimal</h3>
                <p>Best possible (no constraints) for comparison</p>
                <div class="crossings middle">{stats['unconstrained_crosses']:,}</div>
                <div style="color: #666; font-size: 12px;">crossings</div>
                <a href="unconstrained_optimal.html" target="_blank">View Visualization →</a>
            </div>
        </div>
        
        <div class="boundary-info">
            <strong>🚧 BOUNDARY CROSSINGS</strong>
            <br>
            <p style="margin: 10px 0;">
                The boundary between hyperparameters and metrics has 
                <strong style="font-size: 24px;">{stats['boundary_crosses']:,}</strong> crossings.
            </p>
            <p style="font-size: 14px; color: #666;">
                This represents the complexity of the input-output relationship.
                Lower = clearer patterns between hyperparams and performance.
            </p>
        </div>
        
        <div class="dimension-list">
            <h3>📋 Selected Dimensions</h3>
            <div style="margin-top: 20px;">
                <div class="dimension-section hyperparams-section">
                    <strong>⚙️ HYPERPARAMETERS ({len(stats['hyperparams'])})</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        {''.join(f'<li>{hp}</li>' for hp in stats['hyperparams'])}
                    </ul>
                </div>
                
                <div class="dimension-section metrics-section">
                    <strong>📈 METRICS ({len(stats['metrics'])})</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        {''.join(f'<li>{m}</li>' for m in stats['metrics'])}
                    </ul>
                </div>
            </div>
            <p style="margin-top: 20px;">
                <strong>Color metric:</strong> {stats['color_metric']}
            </p>
        </div>
        
        <div style="margin-top: 40px; padding: 25px; background: #e7f3ff; border-radius: 8px;">
            <h3>💡 How to Use These Results</h3>
            <ol>
                <li><strong>Start with Constrained Optimal:</strong> This preserves logical flow and is easiest to interpret</li>
                <li><strong>Check Boundary Crossings:</strong> High crossings = complex input-output relationships</li>
                <li><strong>Compare with Unconstrained:</strong> See the trade-off between optimality and interpretability</li>
                <li><strong>Follow Colored Lines:</strong> Trace high-performing experiments from hyperparams to metrics</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Simple Constrained Parallel Coordinates - YOU choose columns!'
    )
    parser.add_argument('--data', type=str, default='../data/data.csv')
    parser.add_argument('--output', type=str, default='output_constrained')
    parser.add_argument('--algorithm', type=str, default='hybrid',
                       choices=['greedy', 'simulated-annealing', 'hybrid'])
    
    args = parser.parse_args()
    
    # Load data
    print("\n📂 Loading data...")
    data = pd.read_csv(args.data)
    print(f"   Loaded {len(data)} records")
    
    # Column selection - SIMPLE, NO AUTO-DETECTION
    columns_info = select_columns_simple(data)
    hyperparams = columns_info['hyperparams']
    metrics = columns_info['metrics']
    color_col = columns_info['color']
    data_encoded = columns_info['data_encoded']
    categorical_mapping = columns_info['categorical_mapping']
    
    if not hyperparams or not metrics:
        print("\n❌ Error: Must select at least one hyperparam and one metric!")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize constrained optimizer
    print("\n" + "="*70)
    print("⚡ RUNNING CONSTRAINED OPTIMIZATION")
    print("="*70)
    
    start_time = time.time()
    
    # Use encoded data for optimization
    optimizer = ConstrainedParallelCoordinatesOptimizer(data_encoded, hyperparams, metrics)
    optimizer.calculate_all_crossings_fast()
    
    # Compute constrained ordering
    if args.algorithm == 'greedy':
        constrained_order = optimizer.compute_optimal_order_constrained_greedy()
    elif args.algorithm == 'simulated-annealing':
        constrained_order = optimizer.compute_optimal_order_constrained_sa()
    else:  # hybrid
        constrained_order = optimizer.compute_optimal_order_constrained_hybrid()
    
    # Also compute unconstrained for comparison
    from parallel_coordinates_optimized import OptimizedParallelCoordinatesOptimizer
    
    print("\n" + "-"*70)
    print("UNCONSTRAINED OPTIMIZATION (for comparison)")
    print("-"*70)
    
    opt_unconstrained = OptimizedParallelCoordinatesOptimizer(data_encoded, hyperparams + metrics)
    opt_unconstrained.edge_crossings = optimizer.edge_crossings.copy()
    opt_unconstrained.crossing_counts = optimizer.crossing_counts.copy()
    unconstrained_order = opt_unconstrained.compute_optimal_order_greedy()
    
    # Original order
    original_order = hyperparams + metrics
    
    # Calculate crossings
    constrained_crosses = optimizer.get_total_crossings(constrained_order)
    unconstrained_crosses = optimizer.get_total_crossings(unconstrained_order)
    original_crosses = optimizer.get_total_crossings(original_order)
    boundary_crosses = optimizer.get_boundary_crossings(constrained_order)
    
    total_time = time.time() - start_time
    
    improvement = ((original_crosses - constrained_crosses) / original_crosses * 100) if original_crosses > 0 else 0
    
    print("\n" + "="*70)
    print("📊 RESULTS COMPARISON")
    print("="*70)
    print(f"Original order:        {original_crosses:,} crossings")
    print(f"Constrained optimal:   {constrained_crosses:,} crossings ({improvement:+.1f}%)")
    print(f"Unconstrained optimal: {unconstrained_crosses:,} crossings")
    print(f"Boundary crossings:    {boundary_crosses:,}")
    print(f"Total time:            {total_time:.3f}s")
    print("="*70)
    
    # Verify constraint
    hyperparam_indices = [constrained_order.index(h) for h in hyperparams]
    metric_indices = [constrained_order.index(m) for m in metrics]
    
    if max(hyperparam_indices) < min(metric_indices):
        print("✅ CONSTRAINT VERIFIED: All hyperparams before all metrics")
    else:
        print("❌ WARNING: Constraint violated!")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    boundary_idx = len(hyperparams) - 1
    
    print("   1/3 Constrained optimal...")
    fig_constrained = create_parallel_coordinates(
        data_encoded, constrained_order,
        title=f"✅ Constrained Optimal ({constrained_crosses:,} crossings) | HYPERPARAMS → METRICS",
        color_col=color_col,
        boundary_idx=boundary_idx,
        width=1400,
        height=700
    )
    fig_constrained.write_html(os.path.join(args.output, 'constrained_optimal.html'))
    
    print("   2/3 Original order...")
    fig_original = create_parallel_coordinates(
        data_encoded, original_order,
        title=f"📊 Original Order ({original_crosses:,} crossings)",
        color_col=color_col,
        boundary_idx=boundary_idx,
        width=1400,
        height=700
    )
    fig_original.write_html(os.path.join(args.output, 'constrained_original.html'))
    
    print("   3/3 Unconstrained optimal...")
    fig_unconstrained = create_parallel_coordinates(
        data_encoded, unconstrained_order,
        title=f"🔓 Unconstrained Optimal ({unconstrained_crosses:,} crossings) | Mixed Order",
        color_col=color_col,
        width=1400,
        height=700
    )
    fig_unconstrained.write_html(os.path.join(args.output, 'unconstrained_optimal.html'))
    
    # Create summary
    print("   Creating summary page...")
    stats = {
        'n_records': len(data),
        'hyperparams': hyperparams,
        'metrics': metrics,
        'color_metric': color_col,
        'constrained_crosses': constrained_crosses,
        'unconstrained_crosses': unconstrained_crosses,
        'original': original_crosses,
        'boundary_crosses': boundary_crosses,
        'improvement': improvement,
        'time': total_time
    }
    
    summary_path = os.path.join(args.output, 'index.html')
    create_constrained_summary(summary_path, stats)
    
    print("\n" + "="*70)
    print("✨ CONSTRAINED VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output: {args.output}/")
    print(f"🌐 Opening: {summary_path}")
    
    webbrowser.open('file://' + os.path.abspath(summary_path))
    
    print("\n💡 Key Insight:")
    print(f"   Constrained: {constrained_crosses:,} crossings (interpretable)")
    print(f"   Unconstrained: {unconstrained_crosses:,} crossings (optimal)")
    print(f"   Trade-off: {constrained_crosses - unconstrained_crosses:,} extra crossings for logical flow")
    
    print("\n✅ Done! Check the summary for detailed comparison.")


if __name__ == '__main__':
    main()
