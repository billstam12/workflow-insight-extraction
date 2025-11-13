"""
Constrained Parallel Coordinates Optimizer
Enforces hyperparameters always come before metrics (inputs → outputs ordering)
Uses optimized O(n log n) sweep line algorithm
"""

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from typing import List, Tuple, Dict, Set
import plotly.graph_objects as go
import random
import math


class ConstrainedParallelCoordinatesOptimizer:
    """
    Optimizes axis ordering with constraint: hyperparameters before metrics.
    This respects the logical flow: inputs (hyperparams) → outputs (metrics)
    """
    
    def __init__(self, data: pd.DataFrame, hyperparams: List[str], metrics: List[str]):
        """
        Initialize with separate hyperparam and metric lists.
        
        Args:
            data: DataFrame containing the data
            hyperparams: List of hyperparameter column names
            metrics: List of metric column names
        """
        self.data = data
        self.hyperparams = hyperparams
        self.metrics = metrics
        self.all_columns = hyperparams + metrics
        self.edge_crossings = {}
        self.crossing_counts = []
        
    def calculate_edge_crossings_sweep(self, col1: str, col2: str) -> int:
        """
        Calculate edge crossings using O(n log n) sweep line algorithm.
        """
        n = len(self.data)
        values1 = self.data[col1].values
        values2 = self.data[col2].values
        
        # Create lines and sort by first coordinate
        lines = [(values1[i], values2[i], i) for i in range(n)]
        lines.sort(key=lambda x: (x[0], x[1]))
        
        # Extract second coordinates
        second_coords = [line[1] for line in lines]
        
        # Count inversions
        return self._count_inversions(second_coords)
    
    def _count_inversions(self, arr: List[float]) -> int:
        """Count inversions using merge sort."""
        if len(arr) <= 1:
            return 0
        return self._merge_sort_count(arr.copy())[1]
    
    def _merge_sort_count(self, arr: List[float]) -> Tuple[List[float], int]:
        """Merge sort that counts inversions."""
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inv = self._merge_sort_count(arr[:mid])
        right, right_inv = self._merge_sort_count(arr[mid:])
        merged, split_inv = self._merge_count(left, right)
        
        return merged, left_inv + right_inv + split_inv
    
    def _merge_count(self, left: List[float], right: List[float]) -> Tuple[List[float], int]:
        """Merge two sorted arrays and count split inversions."""
        merged = []
        inversions = 0
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i
                j += 1
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions
    
    def calculate_all_crossings_fast(self) -> Dict[Tuple[str, str], int]:
        """Calculate edge crossings for all pairs using sweep line."""
        print("Calculating edge crossings (constrained optimization)...")
        
        total_pairs = len(list(combinations(self.all_columns, 2)))
        computed = 0
        
        for col1, col2 in combinations(self.all_columns, 2):
            crosses = self.calculate_edge_crossings_sweep(col1, col2)
            self.edge_crossings[(col1, col2)] = crosses
            self.crossing_counts.append({
                'axes': (col1, col2),
                'crosses': crosses
            })
            
            computed += 1
            if computed % 10 == 0 or computed == total_pairs:
                print(f"  Progress: {computed}/{total_pairs} pairs ({100*computed/total_pairs:.1f}%)")
        
        self.crossing_counts.sort(key=lambda x: x['crosses'])
        print(f"✓ Calculated crossings for {len(self.crossing_counts)} axis pairs")
        return self.edge_crossings
    
    def get_total_crossings(self, order: List[str]) -> int:
        """Calculate total crossings for given ordering."""
        total = 0
        for i in range(len(order) - 1):
            pair = tuple(sorted([order[i], order[i + 1]]))
            if pair in self.edge_crossings:
                total += self.edge_crossings[pair]
            elif (pair[1], pair[0]) in self.edge_crossings:
                total += self.edge_crossings[(pair[1], pair[0])]
        return total
    
    def compute_optimal_order_constrained_greedy(self) -> List[str]:
        """
        Optimize within constraint: hyperparams first, then metrics.
        Applies greedy clustering separately to each group, then concatenates.
        """
        if not self.crossing_counts:
            self.calculate_all_crossings_fast()
        
        print("\nComputing constrained optimal ordering (greedy clustering)...")
        print(f"  Constraint: {len(self.hyperparams)} hyperparams → {len(self.metrics)} metrics")
        
        # Optimize hyperparameters ordering
        hyperparam_order = self._cluster_subset(self.hyperparams, "hyperparameters")
        
        # Optimize metrics ordering
        metric_order = self._cluster_subset(self.metrics, "metrics")
        
        # Combine: hyperparams first, then metrics
        final_order = hyperparam_order + metric_order
        
        print(f"✓ Constrained ordering computed: {len(final_order)} axes")
        return final_order
    
    def _cluster_subset(self, columns: List[str], name: str) -> List[str]:
        """Apply greedy clustering to a subset of columns."""
        if len(columns) <= 1:
            return columns
        
        print(f"  Optimizing {name} ({len(columns)} dimensions)...")
        
        # Get crossings only within this subset
        subset_crossings = [
            pair for pair in self.crossing_counts 
            if pair['axes'][0] in columns and pair['axes'][1] in columns
        ]
        
        # Initialize clusters
        clusters = [[col] for col in columns]
        axis_to_cluster = {col: i for i, col in enumerate(columns)}
        
        # Greedy clustering
        for pair_info in subset_crossings:
            axis1, axis2 = pair_info['axes']
            
            ends = self._get_cluster_ends(clusters)
            i = axis_to_cluster[axis1]
            j = axis_to_cluster[axis2]
            
            if axis1 in ends and axis2 in ends and i != j:
                ci = clusters[i]
                cj = clusters[j]
                
                # Update cluster assignments
                for a in cj:
                    axis_to_cluster[a] = i
                
                clusters[j] = []
                
                # Join clusters
                if ci[0] == axis1 and cj[0] == axis2:
                    clusters[i] = list(reversed(cj)) + ci
                elif ci[-1] == axis1 and cj[0] == axis2:
                    clusters[i] = ci + cj
                elif ci[0] == axis1 and cj[-1] == axis2:
                    clusters[i] = cj + ci
                else:
                    clusters[i] = ci + list(reversed(cj))
                
                if len(clusters[i]) == len(columns):
                    return clusters[i]
        
        # Flatten clusters
        result = [axis for cluster in clusters for axis in cluster if axis]
        return result
    
    def compute_optimal_order_constrained_sa(self, 
                                            initial_temp: float = 50.0,
                                            cooling_rate: float = 0.95,
                                            iterations: int = 500) -> List[str]:
        """
        Constrained simulated annealing: swaps only within same group.
        """
        if not self.crossing_counts:
            self.calculate_all_crossings_fast()
        
        print("\nComputing constrained optimal ordering (simulated annealing)...")
        print(f"  Initial temp: {initial_temp}, cooling: {cooling_rate}")
        
        # Start with greedy solution
        current_order = self.compute_optimal_order_constrained_greedy()
        current_cost = self.get_total_crossings(current_order)
        
        best_order = current_order.copy()
        best_cost = current_cost
        
        temp = initial_temp
        step = 0
        improvements = 0
        
        n_hyperparams = len(self.hyperparams)
        
        while temp > 0.1:
            for _ in range(iterations):
                # Randomly choose to swap within hyperparams or metrics
                if random.random() < 0.5 and n_hyperparams > 1:
                    # Swap within hyperparameters
                    i = random.randint(0, n_hyperparams - 2)
                    new_order = current_order.copy()
                    new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                elif len(self.metrics) > 1:
                    # Swap within metrics
                    i = random.randint(n_hyperparams, len(current_order) - 2)
                    new_order = current_order.copy()
                    new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                else:
                    continue
                
                new_cost = self.get_total_crossings(new_order)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_order = new_order
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_order = current_order.copy()
                        best_cost = current_cost
                        improvements += 1
                
                step += 1
            
            temp *= cooling_rate
            
            if step % (iterations * 5) == 0:
                print(f"  Step {step}: Best cost = {best_cost:,}, Temp = {temp:.2f}")
        
        print(f"✓ Constrained annealing complete: {improvements} improvements")
        print(f"  Final cost: {best_cost:,} crossings")
        
        return best_order
    
    def compute_optimal_order_constrained_hybrid(self) -> List[str]:
        """
        Hybrid constrained optimization.
        """
        print("\n" + "="*70)
        print("CONSTRAINED HYBRID OPTIMIZATION")
        print("="*70)
        print(f"Constraint: {len(self.hyperparams)} hyperparams → {len(self.metrics)} metrics")
        
        # Stage 1: Greedy
        print("\nStage 1/2: Constrained greedy clustering...")
        greedy_order = self.compute_optimal_order_constrained_greedy()
        greedy_cost = self.get_total_crossings(greedy_order)
        print(f"  Greedy result: {greedy_cost:,} crossings")
        
        # Stage 2: Simulated annealing
        print("\nStage 2/2: Constrained simulated annealing...")
        sa_order = self.compute_optimal_order_constrained_sa(
            initial_temp=50.0,
            cooling_rate=0.95,
            iterations=500
        )
        sa_cost = self.get_total_crossings(sa_order)
        print(f"  Annealing result: {sa_cost:,} crossings")
        
        # Summary
        print("\n" + "="*70)
        print("CONSTRAINED OPTIMIZATION SUMMARY")
        print("="*70)
        print(f"Greedy:     {greedy_cost:,} crossings")
        print(f"Annealing:  {sa_cost:,} crossings ({100*(greedy_cost-sa_cost)/greedy_cost:+.1f}%)")
        print("="*70)
        
        return sa_order if sa_cost < greedy_cost else greedy_order
    
    def _get_cluster_ends(self, clusters: List[List[str]]) -> Set[str]:
        """Get all axes at cluster endpoints."""
        ends = set()
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            ends.add(cluster[0])
            if len(cluster) > 1:
                ends.add(cluster[-1])
        return ends
    
    def get_boundary_crossings(self, order: List[str]) -> int:
        """
        Calculate crossings at the boundary between hyperparams and metrics.
        This is where most visual complexity occurs.
        """
        boundary_idx = len(self.hyperparams) - 1
        if boundary_idx < 0 or boundary_idx >= len(order) - 1:
            return 0
        
        col1 = order[boundary_idx]
        col2 = order[boundary_idx + 1]
        pair = tuple(sorted([col1, col2]))
        
        if pair in self.edge_crossings:
            return self.edge_crossings[pair]
        return 0


def create_parallel_coordinates(data: pd.DataFrame, 
                                columns: List[str], 
                                title: str = "Parallel Coordinates",
                                color_col: str = None,
                                boundary_idx: int = None,
                                width: int = 1400,
                                height: int = 700) -> go.Figure:
    """
    Create parallel coordinates with visual boundary between hyperparams and metrics.
    
    Args:
        boundary_idx: Index where hyperparams end and metrics begin
    """
    dimensions = []
    
    for i, col in enumerate(columns):
        dim_dict = dict(
            label=col,
            values=data[col],
            range=[data[col].min(), data[col].max()]
        )
        dimensions.append(dim_dict)
    
    # Determine line colors
    if color_col and color_col in data.columns:
        line_color = data[color_col]
        colorscale = 'Viridis'
    else:
        metric_cols = [c for c in columns if 'accuracy' in c.lower() or 'f1' in c.lower()]
        if metric_cols:
            line_color = data[metric_cols[0]]
            colorscale = 'Viridis'
        else:
            line_color = data[columns[0]]
            colorscale = 'Blues'
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=line_color,
                colorscale=colorscale,
                showscale=True,
                cmin=line_color.min(),
                cmax=line_color.max()
            ),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        font=dict(size=10)
    )
    
    # Add boundary annotation if provided
    if boundary_idx is not None:
        fig.add_annotation(
            x=(boundary_idx + 0.5) / len(columns),
            y=1.05,
            xref="paper",
            yref="paper",
            text="⬇ HYPERPARAMS | METRICS ⬇",
            showarrow=False,
            font=dict(size=12, color="red", family="Arial Black"),
            bgcolor="rgba(255, 255, 0, 0.3)",
            bordercolor="red",
            borderwidth=2,
            borderpad=5
        )
    
    return fig


if __name__ == "__main__":
    # Test constrained optimization
    print("Constrained Parallel Coordinates Optimizer")
    print("=" * 70)
    
    data = pd.read_csv('../data/data.csv')
    print(f"Loaded {len(data)} records")
    
    # Define hyperparams and metrics
    hyperparams = ['split_ratio', 'max_depth', 'learning_rate', 'n_estimators']
    metrics = ['I2CAT_accuracy', 'I2CAT_f1score', 'I2CAT_recall']
    
    print(f"\nHyperparameters: {hyperparams}")
    print(f"Metrics: {metrics}")
    
    # Test constrained optimizer
    optimizer = ConstrainedParallelCoordinatesOptimizer(data, hyperparams, metrics)
    best_order = optimizer.compute_optimal_order_constrained_hybrid()
    
    print(f"\n✓ Final ordering: {best_order}")
    
    # Verify constraint
    hyperparam_indices = [best_order.index(h) for h in hyperparams]
    metric_indices = [best_order.index(m) for m in metrics]
    
    if max(hyperparam_indices) < min(metric_indices):
        print("✅ CONSTRAINT SATISFIED: All hyperparams before all metrics")
    else:
        print("❌ CONSTRAINT VIOLATED")
