import warnings
import numpy as np
import pandas as pd
import os
import psutil
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
from itertools import product


warnings.filterwarnings("ignore")


# Split
data = pd.read_parquet('taxi.parquet')
data = data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', "store_and_fwd_flag"], axis=1)
data = data.dropna()

frac = 0.005
# Downsample to frac% of the cleaned dataset to reduce runtime
data = data.sample(frac=frac, random_state=42).reset_index(drop=True)
print(f"Using {frac * 100}% sample: {len(data)} rows")
train, test = train_test_split(data, test_size=0.25, random_state=42)

# Separate features and target
train_x = train.drop(["fare_amount"], axis=1)
test_x = test.drop(["fare_amount"], axis=1)
train_y = train[["fare_amount"]]
test_y = test[["fare_amount"]]

print(f"Training set size: {train_x.shape[0]}")
print(f"Test set size: {test_x.shape[0]}")


# Transformation
numeric_cols = train_x.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in train_x.columns if c not in numeric_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)

# Validation hyperparameters (RandomForest) - Expanded for ~600 workflows
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}
# Total combinations: 6 × 6 × 3 × 2 × 3 = 648 workflows
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split'],
    param_grid['min_samples_leaf'],
    param_grid['max_features']
))

print(f"Total number of hyperparameter combinations: {len(param_combinations)}")
print("\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

results_list = []
start_time = time.time()
print(f"Starting {len(param_combinations)} experiments...")
print("="*70)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline
dummy = DummyRegressor(strategy="mean")
dummy.fit(train_x, train_y.values.ravel())
baseline_pred_test = dummy.predict(test_x)
baseline_rmse = float(np.sqrt(mean_squared_error(test_y, baseline_pred_test)))
baseline_mae = float(mean_absolute_error(test_y, baseline_pred_test))
baseline_r2 = float(r2_score(test_y, baseline_pred_test))

# Training + Evaluation per configuration
for idx, params in enumerate(param_combinations, 1):
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params

    # Progress: start of experiment
    print(f"[Exp {idx}/{len(param_combinations)}] Start | "
          f"n_estimators={n_estimators}, max_depth={max_depth}, "
          f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
          f"max_features={max_features}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # System metrics before training
    sys_cpu_before = psutil.cpu_percent(interval=0.1)
    sys_mem_before = psutil.virtual_memory().percent
    sys_mem_used_gb_before = psutil.virtual_memory().used / (1024**3)

    train_start = time.time()
    pipeline.fit(train_x, train_y.values.ravel())
    train_time = time.time() - train_start

    # System metrics after training
    sys_cpu_after = psutil.cpu_percent(interval=0.1)
    sys_mem_after = psutil.virtual_memory().percent
    sys_mem_used_gb_after = psutil.virtual_memory().used / (1024**3)
    sys_mem_peak_gb = max(sys_mem_used_gb_before, sys_mem_used_gb_after)

    pred_train = pipeline.predict(train_x)
    pred_test = pipeline.predict(test_x)

    # Evaluation metrics
    test_rmse = float(np.sqrt(mean_squared_error(test_y, pred_test)))
    test_mae = float(mean_absolute_error(test_y, pred_test))
    test_r2 = float(r2_score(test_y, pred_test))
    test_mse = float(mean_squared_error(test_y, pred_test))
    test_explained_var = float(explained_variance_score(test_y, pred_test))
    test_max_error = float(max_error(test_y, pred_test))
    test_mape = float(mean_absolute_percentage_error(test_y, pred_test))
    test_median_ae = float(median_absolute_error(test_y, pred_test))

    train_rmse = float(np.sqrt(mean_squared_error(train_y, pred_train)))
    train_mae = float(mean_absolute_error(train_y, pred_train))
    train_r2 = float(r2_score(train_y, pred_train))
    train_mse = float(mean_squared_error(train_y, pred_train))

    # Validation metrics (CV)
    cv_neg_mse = cross_val_score(pipeline, train_x, train_y.values.ravel(), cv=kfold, scoring="neg_mean_squared_error")
    cv_neg_mae = cross_val_score(pipeline, train_x, train_y.values.ravel(), cv=kfold, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, train_x, train_y.values.ravel(), cv=kfold, scoring="r2")
    cv_mse_mean = float(-cv_neg_mse.mean())
    cv_rmse_mean = float(np.sqrt(cv_mse_mean))
    cv_mae_mean = float(-cv_neg_mae.mean())
    cv_r2_mean = float(cv_r2.mean())

    # Model complexity summary (feature importances)
    importances = pipeline.named_steps["model"].feature_importances_
    fi_mean = float(np.mean(importances))
    fi_std = float(np.std(importances))
    fi_max = float(np.max(importances))

    # System metrics summary
    sys_cpu_avg = (sys_cpu_before + sys_cpu_after) / 2
    sys_mem_avg = (sys_mem_before + sys_mem_after) / 2
    sys_mem_delta_gb = sys_mem_used_gb_after - sys_mem_used_gb_before

    result = {
        'workflowId': idx,

        # Hyperparameters (training config)
        'n_estimators': n_estimators,
        'max_depth': max_depth if max_depth is not None else -1,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,

        # Test metrics (evaluation)
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_explained_variance': test_explained_var,
        'test_max_error': test_max_error,
        'test_mape': test_mape,
        'test_median_absolute_error': test_median_ae,

        # Train metrics (overfitting indicators)
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mse': train_mse,

        # Model metrics
        'feature_importance_mean': fi_mean,
        'feature_importance_std': fi_std,
        'feature_importance_max': fi_max,
        'training_time_seconds': train_time,

        # Validation (CV)
        'cv_rmse_mean': cv_rmse_mean,
        'cv_mae_mean': cv_mae_mean,
        'cv_r2_mean': cv_r2_mean,
        'cv_mse_mean': cv_mse_mean,

        # Baseline comparison
        'baseline_rmse_mean': baseline_rmse,
        'baseline_mae_mean': baseline_mae,
        'baseline_r2_mean': baseline_r2,

        # Overfitting indicators
        'r2_diff_train_test': train_r2 - test_r2,
        'rmse_diff_train_test': train_rmse - test_rmse,

        # System metrics
        'sys_cpu_percent_avg': sys_cpu_avg,
        'sys_mem_percent_avg': sys_mem_avg,
        'sys_mem_used_gb_before': sys_mem_used_gb_before,
        'sys_mem_used_gb_after': sys_mem_used_gb_after,
        'sys_mem_peak_gb': sys_mem_peak_gb,
        'sys_mem_delta_gb': sys_mem_delta_gb,
    }

    results_list.append(result)

    # Progress: end of experiment summary
    elapsed = time.time() - start_time
    avg_time = elapsed / idx
    remaining = avg_time * (len(param_combinations) - idx)
    best_r2 = max([r['test_r2'] for r in results_list]) if results_list else float('nan')
    print(f"[Exp {idx}/{len(param_combinations)}] Done | "
          f"train_rmse={train_rmse:.3f}, test_rmse={test_rmse:.3f}, test_r2={test_r2:.4f} | "
          f"cv_rmse_mean={cv_rmse_mean:.3f}, cv_r2_mean={cv_r2_mean:.4f} | "
          f"elapsed={elapsed:.1f}s, ETA={remaining:.1f}s, best_r2={best_r2:.4f}")

total_time = time.time() - start_time
print("="*70)
print(f"Completed all {len(param_combinations)} experiments in {total_time:.2f} seconds")
print(f"Average time per run: {total_time/len(param_combinations):.3f} seconds")

output_dir = '../data/workflows_taxi'
os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame(results_list)
output_file = os.path.join(output_dir, 'workflows.csv')
results_df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(f"Shape: {results_df.shape}")
print(f"\nColumns: {list(results_df.columns)}")
print(f"\nFirst few rows:")
print(results_df.head())

param_names = list(param_grid.keys())
param_file = os.path.join(output_dir, 'parameter_names.txt')
with open(param_file, 'w') as f:
    f.write('\n'.join(param_names))
print(f"\nParameter names saved to: {param_file}")

metric_names = [
    'test_rmse', 'test_mae', 'test_r2', 'test_mse',
    'test_explained_variance', 'test_max_error', 'test_mape',
    'test_median_absolute_error', 'train_rmse', 'train_mae',
    'train_r2', 'train_mse', 'feature_importance_mean',
    'feature_importance_std', 'feature_importance_max', 'training_time_seconds',
    'cv_rmse_mean', 'cv_mae_mean', 'cv_r2_mean', 'cv_mse_mean',
    'baseline_rmse_mean', 'baseline_mae_mean', 'baseline_r2_mean',
    'r2_diff_train_test', 'rmse_diff_train_test',
    'sys_cpu_percent_avg', 'sys_mem_percent_avg', 'sys_mem_used_gb_before',
    'sys_mem_used_gb_after', 'sys_mem_peak_gb', 'sys_mem_delta_gb'
]
metric_file = os.path.join(output_dir, 'metric_names.txt')
with open(metric_file, 'w') as f:
    f.write('\n'.join(metric_names))
print(f"Metric names saved to: {metric_file}")

