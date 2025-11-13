"""
Optimized Parallel Coordinates with Advanced Algorithms
- O(n log n) sweep line for edge crossing calculation
- Simulated annealing for global optimization
- Multiple ordering strategies
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Set
import plotly.graph_objects as go
import random
import math


class OptimizedParallelCoordinatesOptimizer:
    """
    High-performance optimizer using advanced algorithms:
    1. Sweep line algorithm: O(n log n) crossing calculation
    2. Simulated annealing: Better than greedy clustering
    3. Two-opt local search: Fast local improvements
    """
    
    def __init__(self, data: pd.DataFrame, columns: List[str]):
        """
        Initialize the optimizer.
        
        Args:
            data: DataFrame containing the data to visualize
            columns: List of column names to include in visualization
        """
        self.data = data
        self.columns = columns
        self.edge_crossings = {}
        self.crossing_counts = []
        
    def calculate_edge_crossings_sweep(self, col1: str, col2: str) -> int:
        """
        Calculate edge crossings using sweep line algorithm.
        Time complexity: O(n log n) vs O(n²) for brute force
        
        The algorithm:
        1. Create events for line endpoints sorted by position
        2. Sweep through events, maintaining active set
        3. Count inversions using modified merge sort
        
        Args:
            col1: First column name
            col2: Second column name
            
        Returns:
            Number of edge crossings
        """
        n = len(self.data)
        
        # Get values and create line segments
        values1 = self.data[col1].values
        values2 = self.data[col2].values
        
        # Create list of (start_rank, end_rank) pairs
        # Rank normalization handles duplicates better
        lines = []
        for i in range(n):
            lines.append((values1[i], values2[i], i))
        
        # Sort by first coordinate
        lines.sort(key=lambda x: (x[0], x[1]))
        
        # Extract second coordinates in sorted order
        second_coords = [line[1] for line in lines]
        
        # Count inversions using merge sort (crossings = inversions)
        return self._count_inversions(second_coords)
    
    def _count_inversions(self, arr: List[float]) -> int:
        """
        Count inversions in array using merge sort.
        An inversion is a pair (i, j) where i < j but arr[i] > arr[j]
        Time complexity: O(n log n)
        
        Args:
            arr: Array of values
            
        Returns:
            Number of inversions (crossings)
        """
        if len(arr) <= 1:
            return 0
        
        return self._merge_sort_count(arr.copy())[1]
    
    def _merge_sort_count(self, arr: List[float]) -> Tuple[List[float], int]:
        """
        Merge sort that also counts inversions.
        
        Args:
            arr: Array to sort
            
        Returns:
            Tuple of (sorted_array, inversion_count)
        """
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inv = self._merge_sort_count(arr[:mid])
        right, right_inv = self._merge_sort_count(arr[mid:])
        
        merged, split_inv = self._merge_count(left, right)
        
        return merged, left_inv + right_inv + split_inv
    
    def _merge_count(self, left: List[float], right: List[float]) -> Tuple[List[float], int]:
        """
        Merge two sorted arrays and count split inversions.
        
        Args:
            left: Left sorted array
            right: Right sorted array
            
        Returns:
            Tuple of (merged_array, split_inversions)
        """
        merged = []
        inversions = 0
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                # All remaining elements in left are greater than right[j]
                inversions += len(left) - i
                j += 1
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged, inversions
    
    def calculate_all_crossings_fast(self) -> Dict[Tuple[str, str], int]:
        """
        Calculate edge crossings for all pairs using sweep line.
        Total complexity: O(n² log n) where n is number of dimensions
        Much faster than O(n³) brute force for large datasets
        
        Returns:
            Dictionary mapping axis pairs to crossing counts
        """
        print("Calculating edge crossings (sweep line algorithm)...")
        
        total_pairs = len(list(combinations(self.columns, 2)))
        computed = 0
        
        for col1, col2 in combinations(self.columns, 2):
            crosses = self.calculate_edge_crossings_sweep(col1, col2)
            self.edge_crossings[(col1, col2)] = crosses
            self.crossing_counts.append({
                'axes': (col1, col2),
                'crosses': crosses
            })
            
            computed += 1
            if computed % 10 == 0 or computed == total_pairs:
                print(f"  Progress: {computed}/{total_pairs} pairs ({100*computed/total_pairs:.1f}%)")
        
        # Sort by number of crossings (ascending)
        self.crossing_counts.sort(key=lambda x: x['crosses'])
        
        print(f"✓ Calculated crossings for {len(self.crossing_counts)} axis pairs")
        return self.edge_crossings
    
    def get_total_crossings(self, order: List[str]) -> int:
        """
        Calculate total edge crossings for a given axis ordering.
        Only counts crossings between adjacent axes.
        
        Args:
            order: List of column names in order
            
        Returns:
            Total number of edge crossings
        """
        total = 0
        for i in range(len(order) - 1):
            pair = tuple(sorted([order[i], order[i + 1]]))
            if pair in self.edge_crossings:
                total += self.edge_crossings[pair]
            elif (pair[1], pair[0]) in self.edge_crossings:
                total += self.edge_crossings[(pair[1], pair[0])]
        return total
    
    def compute_optimal_order_greedy(self) -> List[str]:
        """
        Original OSL2 clustering algorithm (baseline).
        Time complexity: O(n²) where n is number of dimensions
        
        Returns:
            Ordered list of column names
        """
        if not self.crossing_counts:
            self.calculate_all_crossings_fast()
        
        print("Computing optimal ordering (greedy clustering)...")
        
        # Initialize: each axis in its own cluster
        clusters = [[col] for col in self.columns]
        axis_to_cluster = {col: i for i, col in enumerate(self.columns)}
        
        # Process pairs in order of increasing crossings
        for pair_info in self.crossing_counts:
            axis1, axis2 = pair_info['axes']
            
            # Get cluster endpoints
            ends = self._get_cluster_ends(clusters)
            
            i = axis_to_cluster[axis1]
            j = axis_to_cluster[axis2]
            
            # Can only join if both are endpoints and in different clusters
            if axis1 in ends and axis2 in ends and i != j:
                ci = clusters[i]
                cj = clusters[j]
                
                # Update cluster assignments
                for a in cj:
                    axis_to_cluster[a] = i
                
                clusters[j] = []
                
                # Join clusters based on which ends match
                if ci[0] == axis1 and cj[0] == axis2:
                    clusters[i] = list(reversed(cj)) + ci
                elif ci[-1] == axis1 and cj[0] == axis2:
                    clusters[i] = ci + cj
                elif ci[0] == axis1 and cj[-1] == axis2:
                    clusters[i] = cj + ci
                else:  # ci[-1] == axis1 and cj[-1] == axis2
                    clusters[i] = ci + list(reversed(cj))
                
                # If we've merged everything, we're done
                if len(clusters[i]) == len(self.columns):
                    return clusters[i]
        
        # Return flattened clusters
        result = [axis for cluster in clusters for axis in cluster if axis]
        return result
    
    def compute_optimal_order_simulated_annealing(self,
                                                  initial_temp: float = 100.0,
                                                  cooling_rate: float = 0.95,
                                                  iterations: int = 1000) -> List[str]:
        """
        Simulated annealing for global optimization.
        Often finds better solutions than greedy clustering.
        
        Algorithm:
        1. Start with random or greedy ordering
        2. Repeatedly propose swaps of adjacent axes
        3. Accept improvements always, accept worse with probability exp(-ΔE/T)
        4. Decrease temperature over time
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature decay rate (0 < rate < 1)
            iterations: Number of iterations per temperature
            
        Returns:
            Optimized axis ordering
        """
        if not self.crossing_counts:
            self.calculate_all_crossings_fast()
        
        print(f"Computing optimal ordering (simulated annealing)...")
        print(f"  Initial temp: {initial_temp}, cooling: {cooling_rate}, iterations: {iterations}")
        
        # Start with greedy solution
        current_order = self.compute_optimal_order_greedy()
        current_cost = self.get_total_crossings(current_order)
        
        best_order = current_order.copy()
        best_cost = current_cost
        
        temp = initial_temp
        step = 0
        improvements = 0
        
        while temp > 0.1:
            for _ in range(iterations):
                # Propose swap of two adjacent axes
                i = random.randint(0, len(current_order) - 2)
                new_order = current_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                
                new_cost = self.get_total_crossings(new_order)
                delta = new_cost - current_cost
                
                # Accept if better, or with probability based on temperature
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_order = new_order
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_order = current_order.copy()
                        best_cost = current_cost
                        improvements += 1
                
                step += 1
            
            # Cool down
            temp *= cooling_rate
            
            if step % (iterations * 5) == 0:
                print(f"  Step {step}: Best cost = {best_cost:,}, Temp = {temp:.2f}")
        
        print(f"✓ Simulated annealing complete: {improvements} improvements found")
        print(f"  Final cost: {best_cost:,} crossings")
        
        return best_order
    
    def compute_optimal_order_two_opt(self, initial_order: List[str] = None,
                                     max_iterations: int = 100) -> List[str]:
        """
        Two-opt local search for fast improvements.
        Repeatedly reverses segments to reduce crossings.
        
        Algorithm:
        1. Start with initial ordering
        2. Try reversing every possible segment [i, j]
        3. Keep reversal if it improves crossings
        4. Repeat until no improvement
        
        Args:
            initial_order: Starting order (if None, use greedy)
            max_iterations: Maximum number of passes
            
        Returns:
            Locally optimal axis ordering
        """
        if not self.crossing_counts:
            self.calculate_all_crossings_fast()
        
        print("Computing optimal ordering (two-opt local search)...")
        
        # Start with greedy or provided order
        if initial_order is None:
            current_order = self.compute_optimal_order_greedy()
        else:
            current_order = initial_order.copy()
        
        current_cost = self.get_total_crossings(current_order)
        improved = True
        iteration = 0
        total_improvements = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try all possible segment reversals
            for i in range(len(current_order) - 1):
                for j in range(i + 2, len(current_order) + 1):
                    # Reverse segment [i, j)
                    new_order = current_order[:i] + current_order[i:j][::-1] + current_order[j:]
                    new_cost = self.get_total_crossings(new_order)
                    
                    if new_cost < current_cost:
                        current_order = new_order
                        current_cost = new_cost
                        improved = True
                        total_improvements += 1
                        break
                
                if improved:
                    break
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Cost = {current_cost:,}")
        
        print(f"✓ Two-opt complete: {total_improvements} improvements, {iteration} iterations")
        print(f"  Final cost: {current_cost:,} crossings")
        
        return current_order
    
    def compute_optimal_order_hybrid(self) -> List[str]:
        """
        Hybrid approach combining multiple algorithms:
        1. Greedy clustering (fast initial solution)
        2. Simulated annealing (global exploration)
        3. Two-opt (local refinement)
        
        Returns:
            Best ordering found
        """
        print("\n" + "="*70)
        print("HYBRID OPTIMIZATION (3-stage)")
        print("="*70)
        
        # Stage 1: Greedy clustering
        print("\nStage 1/3: Greedy clustering...")
        greedy_order = self.compute_optimal_order_greedy()
        greedy_cost = self.get_total_crossings(greedy_order)
        print(f"  Greedy result: {greedy_cost:,} crossings")
        
        # Stage 2: Simulated annealing
        print("\nStage 2/3: Simulated annealing...")
        sa_order = self.compute_optimal_order_simulated_annealing(
            initial_temp=50.0,
            cooling_rate=0.95,
            iterations=500
        )
        sa_cost = self.get_total_crossings(sa_order)
        print(f"  Annealing result: {sa_cost:,} crossings")
        
        # Stage 3: Two-opt refinement
        print("\nStage 3/3: Two-opt refinement...")
        best_order = sa_order if sa_cost < greedy_cost else greedy_order
        final_order = self.compute_optimal_order_two_opt(best_order, max_iterations=50)
        final_cost = self.get_total_crossings(final_order)
        print(f"  Final result: {final_cost:,} crossings")
        
        # Summary
        print("\n" + "="*70)
        print("OPTIMIZATION SUMMARY")
        print("="*70)
        print(f"Greedy:     {greedy_cost:,} crossings")
        print(f"Annealing:  {sa_cost:,} crossings ({100*(greedy_cost-sa_cost)/greedy_cost:+.1f}%)")
        print(f"Two-opt:    {final_cost:,} crossings ({100*(greedy_cost-final_cost)/greedy_cost:+.1f}%)")
        print("="*70)
        
        return final_order
    
    def _get_cluster_ends(self, clusters: List[List[str]]) -> Set[str]:
        """Get all axes that are at the ends of their clusters."""
        ends = set()
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            ends.add(cluster[0])
            if len(cluster) > 1:
                ends.add(cluster[-1])
        return ends


def create_parallel_coordinates(data: pd.DataFrame, 
                                columns: List[str], 
                                title: str = "Parallel Coordinates",
                                color_col: str = None,
                                width: int = 1400,
                                height: int = 700) -> go.Figure:
    """
    Create an interactive parallel coordinates plot.
    
    Args:
        data: DataFrame containing the data
        columns: List of columns to visualize (in order)
        title: Plot title
        color_col: Column to use for line coloring (optional)
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly figure object
    """
    dimensions = []
    
    for col in columns:
        dimensions.append(
            dict(
                label=col,
                values=data[col],
                range=[data[col].min(), data[col].max()]
            )
        )
    
    # Determine line colors
    if color_col and color_col in data.columns:
        line_color = data[color_col]
        colorscale = 'Viridis'
    else:
        # Use first metric column for coloring if available
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
    
    return fig


if __name__ == "__main__":
    # Performance comparison
    print("Optimized Parallel Coordinates")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv('../data/data.csv')
    print(f"Loaded {len(data)} records")
    
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()[:10]
    print(f"Testing with {len(numeric_cols)} dimensions")
    
    # Test optimized version
    print("\n" + "="*70)
    print("TESTING OPTIMIZED ALGORITHMS")
    print("="*70)
    
    optimizer = OptimizedParallelCoordinatesOptimizer(data, numeric_cols)
    
    # Hybrid optimization
    best_order = optimizer.compute_optimal_order_hybrid()
    
    print(f"\n✓ Best ordering: {best_order}")
