# Workflow Insight Extraction

## Quick Start

### 1. Install Dependencies

You have two options for installing dependencies:

#### Option A: Using pip (Recommended)
For a simpler setup with pip and no conda dependency:

```bash
# Create a virtual environment (requires Python 3.12)
python3.12 -m venv env
source env/bin/activate  # On macOS/Linux
# or: env\Scripts\activate  # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Python 3.12+ is required. The packages do not support Python 3.14.

#### Option B: Using Conda 
If you prefer conda and want the exact environment configuration:

```bash
conda env create -f environment.yml
conda activate xxp
```

### 2. Run the Ablation Study

Execute the complete ablation study across all three datasets with all ablation modes:

```bash
./run_ablation_study.sh
```

This script will:
- Run clustering with 4 ablation modes: `full`, `no_variance_filter`, `no_dim_reduction`, `no_iterative_filter`
- Execute the pipeline for all three datasets: `wine`, `adult`, `taxi`
- Generate quality metrics and plots
- Copy representative and explanation quality images to `paper_results/`

### 3. Check Results

Results are organized by dataset and ablation mode in the `results/` directory:

```
results/
├── wine/
│   ├── full/
│   ├── no_variance_filter/
│   ├── no_dim_reduction/
│   └── no_iterative_filter/
├── adult/
│   └── [same structure as wine]
└── taxi/
    └── [same structure as wine]
```

Generated plots and images are saved in:
```
paper_results/
├── cluster_quality_metrics_*.png
├── cluster_respresentative_quality_*.png
└── explanation_quality_*.png
```

## Project Structure

- `1_cluster_workflows.py` - Workflow clustering implementation
- `2_generate_cluster_insights.py` - Generate insights from clusters
- `4_validate_cluster_quality.py` - Validate cluster quality and generate plots
- `run_ablation_study.sh` - Main ablation study script
- `data/` - Input workflow data for each dataset
- `results/` - Output results from ablation study
- `paper_results/` - Final plots for publication

## Requirements

### Dependencies (from requirements.txt)
- **pandas** - Data manipulation
- **scikit-learn** - Clustering and machine learning
- **prince** - Dimensionality reduction
- **shap** - Explainability
- **plotly** - Interactive visualizations
- **matplotlib** - Publication-quality plots
- **seaborn** - Statistical visualizations
- **numpy** - Numerical computing
- **joblib** - Parallel processing
- **kmedoids** - K-medoids clustering

## Conda vs Pip

**Use Pip (requirements.txt) if:**
- You prefer simplicity
- You already have Python set up
- You're working in a CI/CD pipeline

**Use Conda (environment.yml) if:**
- You want the exact same environment as the original
- You need binary package optimization
- You're working on macOS with Apple Silicon

## Troubleshooting

### Module Import Error
If you see `SyntaxError: invalid decimal literal` when running, ensure you're using the latest version of the script.

### Missing Data Files
Ensure the data is in `./data/workflows_wine/`, `./data/workflows_adult/`, and `./data/workflows_taxi/`

### Permission Denied
Make the script executable:
```bash
chmod +x run_ablation_study.sh
```

## Output Description

For each dataset and ablation mode, you'll get:

- `cluster_quality/` - Davies-Bouldin Index, Silhouette Score
- `representative_quality/` - Cross-validation boxplots
- `explanation_quality/` - SHAP-based explanation scores
- `workflow_clustered.csv` - Cluster assignments

## Citation

If you use this code, please cite the original paper.
