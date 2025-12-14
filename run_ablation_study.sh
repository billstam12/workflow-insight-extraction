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

# Array of datasets
datasets=(
    "wine:./data/workflows_wine"
    "adult:./data/workflows_adult"
    "taxi:./data/workflows_taxi"
)

# Array of ablation modes
modes=("no_variance_filter" "no_dim_reduction" "no_iterative_filter" "full" )

echo "=========================================="
echo "Starting Ablation Study for All Datasets"
echo "=========================================="
echo ""

for dataset_config in "${datasets[@]}"; do
    # Parse dataset configuration
    DATASET_NAME="${dataset_config%%:*}"
    WORKFLOW_FOLDER="${dataset_config##*:}"
    
    echo "=========================================="
    echo "Dataset: $DATASET_NAME"
    echo "Workflow Folder: $WORKFLOW_FOLDER"
    echo "=========================================="
    echo ""

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
    python 4_validate_cluster_quality.py "$WORKFLOW_FOLDER" "$DATASET_NAME" "$mode"
    
    echo ""
    echo "✓ Completed: $mode"
    echo ""
done

    echo ""
    echo "✓ Dataset Complete: $DATASET_NAME"
    echo ""
done

echo ""
echo "=========================================="
echo "Generating Paper Results Plots and Images"
echo "=========================================="
echo ""

# Generate plots for all ablation modes
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('validate_quality', '4_validate_cluster_quality.py')
validate_quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_quality)

modes = ['full', 'no_variance_filter', 'no_dim_reduction', 'no_iterative_filter']
for mode in modes:
    validate_quality.plot_datasets_scores(ablation=mode)
validate_quality.copy_quality_images()
"

echo ""
echo "=========================================="
echo "Ablation Study Complete for All Datasets!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - results/wine/{full,no_variance_filter,no_dim_reduction,no_iterative_filter}/"
echo "  - results/adult/{full,no_variance_filter,no_dim_reduction,no_iterative_filter}/"
echo "  - results/taxi/{full,no_variance_filter,no_dim_reduction,no_iterative_filter}/"
echo ""
echo "Plots are saved in:"
echo "  - paper_results/cluster_quality_metrics_*.png"
echo ""
