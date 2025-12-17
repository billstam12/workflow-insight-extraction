#!/bin/bash
 
# Ablation Study Execution Script
# This script runs the complete clustering pipeline for all ablation modes:
# - full (baseline)
# - no_variance_filter
# - no_dim_reduction
# - no_iterative_filter
 
set -e  # Exit on error
 
# Initialize timing data file
TIMING_CSV="ablation_times.csv"
echo "Pipeline,Ablation,Time(sec)" > "$TIMING_CSV"
 
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
    
    # Record start time
    START_TIME=$(date +%s)
    
    echo "Step 1: Clustering workflows..."
    python 1_cluster_workflows.py "$WORKFLOW_FOLDER" "$mode"
    
    echo ""
    echo "Step 2: Generating cluster insights..."
    python 2_generate_cluster_insights.py "$WORKFLOW_FOLDER" "$mode"
    
    echo ""
    echo "Step 3: Validating cluster quality..."
    python 4_validate_cluster_quality.py "$WORKFLOW_FOLDER" "$DATASET_NAME" "$mode"
    
    # Record end time and calculate elapsed time
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    
    # Append to timing CSV
    echo "$DATASET_NAME,$mode,$ELAPSED_TIME" >> "$TIMING_CSV"
    
    echo ""
    echo "✓ Completed: $mode (Time: ${ELAPSED_TIME}s)"
    
    echo ""
    echo "Cleaning generated files from data folder..."
    cd "$WORKFLOW_FOLDER" && find . -type f ! -name 'workflows.csv' ! -name 'metric_names.txt' ! -name 'parameter_names.txt' -delete
    rm -rf csv
    cd - > /dev/null
    echo "✓ Data folder cleaned"
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
echo "Timing data saved in:"
echo "  - $TIMING_CSV"
echo ""
echo "Plots are saved in:"
echo "  - paper_results/cluster_quality_metrics_*.png"
echo ""