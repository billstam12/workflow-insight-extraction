#!/bin/bash

# Ablation Study Execution Script
# This script runs the complete clustering pipeline for all ablation modes:
# - full (baseline)
# - no_variance_filter
# - no_dim_reduction
# - no_iterative_filter

set -e  # Exit on error

# Activate virtual environment
source env/bin/activate

# Configuration
WORKFLOW_FOLDER="./data/workflows_wine"
DATASET_NAME="wine"

echo "=========================================="
echo "Starting Ablation Study"
echo "=========================================="
echo "Workflow Folder: $WORKFLOW_FOLDER"
echo "Dataset Name: $DATASET_NAME"
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
    python 1_cluster_workflows.py "$WORKFLOW_FOLDER" "$mode"
    
    echo ""
    echo "Step 2: Generating cluster insights..."
    python 2_generate_cluster_insights.py "$WORKFLOW_FOLDER" "$mode"
    
    echo ""
    echo "Step 3: Validating cluster quality..."
    python 4_validate_cluster_quality_copy.py "$WORKFLOW_FOLDER" "$DATASET_NAME" "$mode"
    
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
echo "  - results/$DATASET_NAME/full/"
echo "  - results/$DATASET_NAME/no_variance_filter/"
echo "  - results/$DATASET_NAME/no_dim_reduction/"
echo "  - results/$DATASET_NAME/no_iterative_filter/"
echo ""
