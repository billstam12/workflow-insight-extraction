import warnings
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
import time
from itertools import product


warnings.filterwarnings("ignore")


# Load the wine quality dataset

data = pd.read_csv('wine.csv')

# Split the data into training and test sets (75/25 split)
train, test = train_test_split(data, test_size=0.25, random_state=42)

# Separate features and target
# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

print(f"Training set size: {train_x.shape[0]}")
print(f"Test set size: {test_x.shape[0]}")



# Define hyperparameter grid (7 parameters)
param_grid = {
    'alpha': [0.01, 0.03, 0.05, 0.1, 0.15],
    'l1_ratio': [0.01, 0.03, 0.05, 0.1],
    'max_iter': [1000, 5000],
    'tol': [1e-4, 1e-3, 1e-2],
    'selection': ['cyclic', 'random'],
    'fit_intercept': [True, False],
    'normalize': [True]  # normalize is deprecated but we keep it simple
}

# Generate all combinations
param_combinations = list(product(
    param_grid['alpha'],
    param_grid['l1_ratio'],
    param_grid['max_iter'],
    param_grid['tol'],
    param_grid['selection'],
    param_grid['fit_intercept'],
    param_grid['normalize']
))

print(f"Total number of hyperparameter combinations: {len(param_combinations)}")
print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")



# Store all results
results_list = []

# Start timing
start_time = time.time()

print(f"Starting {len(param_combinations)} experiments...")
print("="*70)

# Run experiments
for idx, params in enumerate(param_combinations, 1):
    alpha, l1_ratio, max_iter, tol, selection, fit_intercept, normalize = params
    
    # Train model with current hyperparameters
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        selection=selection,
        fit_intercept=fit_intercept,
        random_state=42
    )
    
    # Fit the model
    train_start = time.time()
    model.fit(train_x, train_y)
    train_time = time.time() - train_start
    
    # Make predictions
    pred_train = model.predict(train_x)
    pred_test = model.predict(test_x)
    
    # Calculate comprehensive metrics (12+ metrics)
    # Test set metrics
    test_rmse = np.sqrt(mean_squared_error(test_y, pred_test))
    test_mae = mean_absolute_error(test_y, pred_test)
    test_r2 = r2_score(test_y, pred_test)
    test_mse = mean_squared_error(test_y, pred_test)
    test_explained_var = explained_variance_score(test_y, pred_test)
    test_max_error = max_error(test_y, pred_test)
    test_mape = mean_absolute_percentage_error(test_y, pred_test)
    test_median_ae = median_absolute_error(test_y, pred_test)
    
    # Training set metrics (for overfitting analysis)
    train_rmse = np.sqrt(mean_squared_error(train_y, pred_train))
    train_mae = mean_absolute_error(train_y, pred_train)
    train_r2 = r2_score(train_y, pred_train)
    train_mse = mean_squared_error(train_y, pred_train)
    
    # Model complexity metrics
    n_nonzero_coefs = np.sum(model.coef_ != 0)
    coef_mean = np.mean(np.abs(model.coef_))
    coef_std = np.std(model.coef_)
    
    # Store results
    result = {
        # Run info
        'workflowId': idx,
        # 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Hyperparameters
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': max_iter,
        'tol': tol,
        'selection': selection,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        
        # Test metrics
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_explained_variance': test_explained_var,
        'test_max_error': test_max_error,
        'test_mape': test_mape,
        'test_median_absolute_error': test_median_ae,
        
        # Train metrics
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mse': train_mse,
        
        # Model metrics
        'n_nonzero_coefficients': n_nonzero_coefs,
        'coefficients_mean': coef_mean,
        'coefficients_std': coef_std,
        'n_iterations': model.n_iter_,
        'training_time_seconds': train_time,
        
        # Overfitting indicators
        'r2_diff_train_test': train_r2 - test_r2,
        'rmse_diff_train_test': train_rmse - test_rmse,
    }
    
    results_list.append(result)
    
    # Print progress every 50 runs
    if idx % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (len(param_combinations) - idx)
        print(f"Run {idx}/{len(param_combinations)} | "
              f"Elapsed: {elapsed:.1f}s | "
              f"ETA: {remaining:.1f}s | "
              f"Best R2: {max([r['test_r2'] for r in results_list]):.4f}")

# Final timing
total_time = time.time() - start_time
print("="*70)
print(f"Completed all {len(param_combinations)} experiments in {total_time:.2f} seconds")
print(f"Average time per run: {total_time/len(param_combinations):.3f} seconds")


# Create output directory if it doesn't exist
output_dir = '../data/workflows_wine'
os.makedirs(output_dir, exist_ok=True)

# Convert results to DataFrame
results_df = pd.DataFrame(results_list)

# Save to CSV
output_file = os.path.join(output_dir, 'workflows.csv')
results_df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(f"Shape: {results_df.shape}")
print(f"\nColumns: {list(results_df.columns)}")
print(f"\nFirst few rows:")
print(results_df.head())

# Save parameter names to txt
param_names = list(param_grid.keys())
param_file = os.path.join(output_dir, 'parameter_names.txt')
with open(param_file, 'w') as f:
    f.write('\n'.join(param_names))
print(f"\nParameter names saved to: {param_file}")

# Save metric names to txt
metric_names = [
    'test_rmse', 'test_mae', 'test_r2', 'test_mse',
    'test_explained_variance', 'test_max_error', 'test_mape',
    'test_median_absolute_error', 'train_rmse', 'train_mae',
    'train_r2', 'train_mse', 'n_nonzero_coefficients',
    'coefficients_mean', 'coefficients_std', 'n_iterations',
    'training_time_seconds', 'r2_diff_train_test', 'rmse_diff_train_test'
]
metric_file = os.path.join(output_dir, 'metric_names.txt')
with open(metric_file, 'w') as f:
    f.write('\n'.join(metric_names))
print(f"Metric names saved to: {metric_file}")

