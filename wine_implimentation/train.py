"""
Binary Wine Classification Pipeline
Task: Predict whether a wine is white or red from its properties
Pipeline: Load → Split → Transform → Train → Evaluate
"""

import warnings
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, brier_score_loss
)
from sklearn.pipeline import Pipeline
import time
from itertools import product
import psutil

warnings.filterwarnings("ignore")


# ============================================================================
# SYSTEM RESOURCE MONITORING
# ============================================================================
def get_system_metrics():
    """Collect current system resource utilization metrics"""
    metrics = {}
    
    # CPU utilization
    metrics['system/cpu_utilization_percentage'] = psutil.cpu_percent(interval=0.1)
    
    # Memory usage
    mem = psutil.virtual_memory()
    metrics['system/system_memory_usage_megabytes'] = mem.used / (1024 * 1024)  # Convert to MB
    metrics['system/system_memory_usage_percentage'] = mem.percent
    
    # Disk usage
    disk = psutil.disk_usage('/')
    metrics['system/disk_usage_megabytes'] = disk.used / (1024 * 1024)  # Convert to MB
    # metrics['system/disk_available_megabytes'] = disk.free / (1024 * 1024)  # Convert to MB
    # metrics['system/disk_usage_percentage'] = disk.percent
    
    # Network I/O
    net = psutil.net_io_counters()
    metrics['system/network_receive_megabytes'] = net.bytes_recv / (1024 * 1024)  # Convert to MB
    metrics['system/network_transmit_megabytes'] = net.bytes_sent / (1024 * 1024)  # Convert to MB
    
    return metrics


# ============================================================================
# STEP 1: LOAD RAW WINE DATA
# ============================================================================
def load_wine_data():
  
    red_wine = pd.read_csv('winequality-red.csv', sep=';')
    red_wine['wine_type'] = 0  # 0 = red
    
    white_wine = pd.read_csv('winequality-white.csv', sep=';')
    white_wine['wine_type'] = 1  # 1 = white
    
    # Combine datasets
    data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    print(f"Total samples: {len(data)}")
    
    # Check for NaN values and drop rows with NaN
    nan_count = data.isnull().sum().sum()
    print(f"NaN values found: {nan_count}")
    
    if nan_count > 0:
        print("Dropping rows with NaN values...")
        data = data.dropna()
        print(f"Samples after dropping NaN: {len(data)}")
    
    print(f"Features: {list(data.columns[:-2])}")  # Exclude quality and wine_type
    
    # Separate features and target
    X = data.drop(['wine_type'], axis=1)
    y = data['wine_type']
    
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution - Red: {(y==0).sum()}, White: {(y==1).sum()}")
    
    return X, y



# ============================================================================
# STEP 2: SPLIT (TRAIN/VALIDATION/TEST)
# ============================================================================
def split_data(X, y, test_size=0.25, val_size=0.25, random_state=42):
    """Split data into train, validation, and test sets"""
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATA (TRAIN/VALIDATION/TEST)")
    print("="*70)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_transformer(transformer_type='standard'):
    """Create and return a feature transformer"""
    transformers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'none': None
    }
    return transformers.get(transformer_type, StandardScaler())


def create_classifier(classifier_type, **params):
    """Create classifier with given hyperparameters"""
    if classifier_type == 'logistic':
        # Handle elasticnet penalty - requires l1_ratio parameter
        if params.get('penalty') == 'elasticnet':
            if 'l1_ratio' not in params:
                params['l1_ratio'] = 0.5  # Default l1_ratio
        # Set default random_state if not provided
        if 'random_state' not in params:
            params['random_state'] = 42
        return LogisticRegression(**params, max_iter=2000)
    elif classifier_type == 'random_forest':
        # random_state is already in params from the grid
        return RandomForestClassifier(**params)
    elif classifier_type == 'svm':
        if 'random_state' not in params:
            params['random_state'] = 42
        return SVC(**params, probability=True)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")



def compute_fairness_metrics(y_true, y_pred, y_proba):
    """Compute fairness metrics treating classes as protected groups"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall for positive)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Acceptance rates
    positive_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0
    negative_rate = (tn + fn) / len(y_true) if len(y_true) > 0 else 0
    
    fairness = {}
    
    # Core Fairness Metrics
    fairness['Statistical_Parity'] = abs(positive_rate - 0.5)  # Deviation from 50-50
    fairness['Disparate_Impact'] = min(positive_rate, negative_rate) / max(positive_rate, negative_rate) if max(positive_rate, negative_rate) > 0 else 1.0
    fairness['Equal_Opportunity'] = abs(tpr - tnr)  # TPR difference between groups
    fairness['Equalized_Odds'] = abs(tpr - tnr) + abs(fpr - (1-tnr))  # Both TPR and FPR equality
    fairness['Predictive_Equality'] = abs(fpr - fnr)  # FPR equality
    fairness['Treatment_Equality'] = abs(fn - fp) / (fn + fp) if (fn + fp) > 0 else 0
    fairness['Equal_Negative_Predictive_Value'] = abs(npv - 0.5)  # NPV balance
    
    # Conditional Metrics
    fairness['Conditional_Statistical_Parity'] = abs(ppv - npv)  # Predictive parity
    fairness['Conditional_Use_Accuracy_Equality'] = abs(ppv - (1-ppv))  # Precision balance
    fairness['Overall_Accuracy_Equality'] = abs(accuracy_score(y_true, y_pred) - balanced_accuracy_score(y_true, y_pred))
    
    # Test Fairness (calibration-based)
    fairness['Test_Fairness'] = abs(np.mean(y_proba[y_true == 1]) - np.mean(y_proba[y_true == 0]))
    
    # Well Calibration (Brier score - lower is better calibration)
    fairness['Well_Calibration'] = brier_score_loss(y_true, y_proba)
    
    # Balance metrics per class
    fairness['Balance_for_Positive_Class'] = tpr  # Recall for positive class
    fairness['Balance_for_Negative_Class'] = tnr  # Recall for negative class
    
    return fairness


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Compute comprehensive performance and fairness metrics"""
    
    # Predictions for all sets
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    metrics = {}
    
    # ===== TEST SET METRICS =====
    # Core Performance Metrics
    metrics['test_AUC_ROC'] = roc_auc_score(y_test, y_test_proba)
    metrics['test_Accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['test_Precision'] = precision_score(y_test, y_test_pred)
    metrics['test_Recall'] = recall_score(y_test, y_test_pred)
    metrics['test_F1_Score'] = f1_score(y_test, y_test_pred)
    metrics['test_Log_Loss'] = log_loss(y_test, y_test_proba)
    metrics['test_Matthews_Corrcoef'] = matthews_corrcoef(y_test, y_test_pred)
    metrics['test_Cohen_Kappa'] = cohen_kappa_score(y_test, y_test_pred)
    metrics['test_Balanced_Accuracy'] = balanced_accuracy_score(y_test, y_test_pred)
    
    # Test Confusion Matrix components
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
    metrics['test_True_Positives'] = tp_test
    metrics['test_True_Negatives'] = tn_test
    metrics['test_False_Positives'] = fp_test
    metrics['test_False_Negatives'] = fn_test
    metrics['test_Specificity'] = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0
    metrics['test_Sensitivity'] = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
    
    # Test Fairness Metrics
    fairness_metrics_test = compute_fairness_metrics(y_test, y_test_pred, y_test_proba)
    for key, value in fairness_metrics_test.items():
        metrics[f'test_{key}'] = value
    
    # Validation set is used internally but metrics not reported to reduce output size
    
    # ===== TRAIN SET METRICS (for overfitting detection) =====
    metrics['train_Accuracy'] = accuracy_score(y_train, y_train_pred)
    metrics['train_F1_Score'] = f1_score(y_train, y_train_pred)
    metrics['train_AUC_ROC'] = roc_auc_score(y_train, y_train_proba)
    metrics['train_Precision'] = precision_score(y_train, y_train_pred)
    metrics['train_Recall'] = recall_score(y_train, y_train_pred)
    
    # Overfitting indicators (Train vs Test)
    metrics['overfitting_Accuracy_diff'] = metrics['train_Accuracy'] - metrics['test_Accuracy']
    metrics['overfitting_F1_diff'] = metrics['train_F1_Score'] - metrics['test_F1_Score']
    metrics['overfitting_AUC_diff'] = metrics['train_AUC_ROC'] - metrics['test_AUC_ROC']
    
    # Generalization gap removed (validation metrics not reported)
    
    # Learning Curves (use subset for speed)
    try:
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, cv=3, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        metrics['learning_curve_train_score'] = np.mean(train_scores[-1])  # Final train score
        metrics['learning_curve_test_score'] = np.mean(test_scores[-1])    # Final test score
    except:
        metrics['learning_curve_train_score'] = np.nan
        metrics['learning_curve_test_score'] = np.nan
    
    return metrics



# Load data
X, y = load_wine_data()

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print("\n" + "="*70)
print("STEP 3-5: TRANSFORM → TRAIN → EVALUATE")
print("="*70)


param_grid = {
    'normalization': ['standard', 'minmax', 'robust'],
    'classifier': ['random_forest'],
    # Random Forest params -
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 10, 20],
    'fairness_method': ["balanced_subsample", "balanced", None],  # Maps to class_weight
    'random_state': [42]  # Fixed for reproducibility
}

# Expected total: 3 normalizations × 6 n_estimators × 3 criterion × 6 max_depth × 2 fairness = ~648 workflows

# Generate combinations based on classifier type
results_list = []
workflow_id = 1

print(f"\nGenerating hyperparameter combinations...")

# Start timing
start_time = time.time()

# Iterate through transformers and classifiers
for normalization in param_grid['normalization']:
    for classifier_type in param_grid['classifier']:
        
        if classifier_type == 'random_forest':
            # Random forest combinations with fairness
            for n_est, criterion, max_d, fairness_method, rand_state in product(
                param_grid['n_estimators'],
                param_grid['criterion'],
                param_grid['max_depth'],
                param_grid['fairness_method'],
                param_grid['random_state']
            ):
                try:
                    # STEP 3: Create transformer
                    transformer = create_transformer(normalization)
                    
                    # STEP 4: Create and train model
                    classifier = create_classifier(
                        classifier_type,
                        n_estimators=n_est,
                        criterion=criterion,
                        max_depth=max_d,
                        class_weight=fairness_method,
                        random_state=rand_state
                    )
                    
                    # Build pipeline
                    if transformer is not None:
                        model = Pipeline([
                            ('scaler', transformer),
                            ('classifier', classifier)
                        ])
                    else:
                        model = Pipeline([('classifier', classifier)])
                    
                    # Train
                    train_start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - train_start
                    
                    # STEP 5: Evaluate
                    metrics = evaluate_model(
                        model, X_train, X_val, X_test,
                        y_train, y_val, y_test
                    )
                    
                    # Collect system metrics
                    system_metrics = get_system_metrics()
                    
                    # Store results (only RF-relevant parameters)
                    result = {
                        'workflowId': workflow_id,
                        'normalization': normalization,
                        'classifier': classifier_type,
                        'n_estimators': n_est,
                        'criterion': criterion,
                        'max_depth': max_d if max_d is not None else -1,
                        'fairness_method': str(fairness_method),
                        'random_state': rand_state,
                        'training_time_seconds': train_time,
                        **metrics,
                        **system_metrics
                    }
                    
                    results_list.append(result)
                    
                    # Print progress with time estimate
                    elapsed = time.time() - start_time
                    avg_time = elapsed / workflow_id
                    print(f"[{workflow_id}] RandomForest | {normalization} | n_est={n_est} | depth={max_d} | criterion={criterion} | "
                          f"acc={metrics['test_Accuracy']:.4f} | time={train_time:.2f}s | avg={avg_time:.2f}s")
                    
                    workflow_id += 1
                        
                except Exception as e:
                    print(f"Skipping combination due to error: {e}")
                    continue



# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create output directory
output_dir = '../data/workflows_wine'
os.makedirs(output_dir, exist_ok=True)

# Convert results to DataFrame
results_df = pd.DataFrame(results_list)

# Save to CSV
output_file = os.path.join(output_dir, 'workflows.csv')
results_df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(f"Shape: {results_df.shape}")
print(f"\nBest models by Test Accuracy:")
print(results_df.nlargest(5, 'test_Accuracy')[['workflowId', 'normalization', 'classifier', 'test_Accuracy', 'test_F1_Score', 'test_AUC_ROC']])

# Save parameter names (Random Forest only)
param_names = ['normalization', 'classifier', 'criterion', 'max_depth', 'n_estimators', 'fairness_method', 'random_state']
param_file = os.path.join(output_dir, 'parameter_names.txt')
with open(param_file, 'w') as f:
    f.write('\n'.join(param_names))
print(f"\nParameter names saved to: {param_file}")

# Save metric names
metric_names = [col for col in results_df.columns 
                if col not in ['workflowId'] + param_names + ['training_time_seconds']]
metric_file = os.path.join(output_dir, 'metric_names.txt')
with open(metric_file, 'w') as f:
    f.write('\n'.join(metric_names))
print(f"Metric names saved to: {metric_file}")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)

