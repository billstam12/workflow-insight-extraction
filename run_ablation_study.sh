#!/bin/bash

# Ablation Study Execution Script
# This script runs the complete clustering pipeline for all ablation modes:
# - full (baseline)
# - no_variance_filter
# - no_dim_reduction
# - no_iterative_filter

set -e  # Exit on error

echo "=========================================="
echo "Starting Ablation Study"
echo "=========================================="
echo ""

# Array of ablation modes
modes=("full" "no_variance_filter" "no_dim_reduction" "no_iterative_filter")

for mode in "${modes[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Pipeline: $mode"
    echo "=========================================="
    echo ""
    
    echo "Step 1: Clustering workflows..."
    python 1_cluster_workflows.py ./data/workflows "$mode"
    
    echo ""
    echo "Step 2: Generating cluster insights..."
    python 2_generate_cluster_insights.py ./data/workflows "$mode"
    
    echo ""
    echo "Step 3: Validating cluster quality..."
    python 4_validate_cluster_quality.py ./data/workflows adult "$mode"
    
    echo ""
    echo "✓ Completed: $mode"
    echo ""
done

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - results/adult/full/"
echo "  - results/adult/no_variance_filter/"
echo "  - results/adult/no_dim_reduction/"
echo "  - results/adult/no_iterative_filter/"
echo ""
