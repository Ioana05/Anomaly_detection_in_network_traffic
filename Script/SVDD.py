from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.deep_svdd import DeepSVDD
from sklearn.model_selection import RandomizedSearchCV # Import this
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, f1_score)
import numpy as np
from load_files import X_res, y_res

def balance(dataset, min_samples=100, max_samples=3000, random_state=42):
    if 'attack_cat' not in dataset.columns:
        raise ValueError("Column 'attack_cat' is required for stratified balancing.")

    normal = dataset[dataset['label'] == 0]
    anomalies = dataset[dataset['label'] == 1]
    attack_types = anomalies['attack_cat'].unique()

    sampled_anomalies = []
    for attack in attack_types:
        subset = anomalies[anomalies['attack_cat'] == attack]
        count = min(max(len(subset), min_samples), max_samples)
        sampled = subset.sample(n=min(count, len(subset)), random_state=random_state)
        sampled_anomalies.append(sampled)

    balanced_anomalies = pd.concat(sampled_anomalies, ignore_index=True)
    balanced_normals = normal.sample(n=len(balanced_anomalies), random_state=random_state, replace = False)

    final = pd.concat([balanced_anomalies, balanced_normals], ignore_index=True).sample(frac=1, random_state=random_state)
    final['id'] = final.index
    print(f"Balanced (semi-stratified) dataset size: {len(final)}")

    return final

def stratified_kfold_balance(dataset, n_splits=5, min_samples=100, max_samples=3000, random_state=42):
    """
    Generate balanced folds while preserving attack type distribution
    Returns: List of (train_idx, test_idx) pairs
    """
    if 'attack_cat' not in dataset.columns:
        raise ValueError("Column 'attack_cat' is required for stratified balancing.")

    # First create balanced dataset using your existing logic
    balanced_data = balance(dataset, min_samples, max_samples, random_state)

    # Create a combined stratification column (attack_cat + label)
    balanced_data['stratify_col'] = balanced_data['attack_cat'].astype(str) + "_" + balanced_data['label'].astype(str)

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Generate folds
    folds = []
    for train_idx, test_idx in skf.split(balanced_data, balanced_data['stratify_col']):
        folds.append((train_idx, test_idx))

    return folds, balanced_data

def SVDD():
    # 1. Generate balanced folds using the resampled data
    folds = stratified_kfold_balance(X_res, y_res, n_splits=5) # Use the resampled data

    n_epochs = 50
    batch_size = 64

    with tqdm(total=len(folds), desc="Training DeepSVDD") as pbar_folds:
        for fold, (train_idx, test_idx) in enumerate(folds):
            print(f"\nTraining Fold {fold + 1}")

            # Get fold data from resampled data
            X_train_fold = X_res[train_idx]
            y_train_fold = y_res[train_idx]
            X_val_fold = X_res[test_idx]
            y_val_fold = y_res[test_idx]

            # Train DeepSVDD model
            n_features = X_train_fold.shape[1]
            model = DeepSVDD(n_features=n_features, epochs=n_epochs, batch_size=batch_size, verbose=0)
            model.fit(X_train_fold)

            # Evaluate
            predictions = model.predict(X_val_fold)
            adjusted_predictions = np.where(predictions == 1, 0, 1)

            f1 = f1_score(y_val_fold, adjusted_predictions)
            acc = accuracy_score(y_val_fold, adjusted_predictions)

            print(f"Fold {fold + 1} F1 Score: {f1:.4f}")
            print(f"Fold {fold + 1} Accuracy score: {acc:.4f}")
            pbar_folds.update(1)

    # Count predictions (assuming this is done after the loop)
    unique, counts = np.unique(predictions, return_counts=True)
    pred_df = pd.DataFrame({'Label': unique, 'Count': counts})

    # 0 = inlier, 1 = outlier
    sns.barplot(data=pred_df, x='Label', y='Count')
    plt.xticks([0, 1], ['Inlier', 'Outlier'])
    plt.title('Prediction Distribution')
    plt.show()

    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    pred_df = pd.DataFrame({'Label': unique, 'Count': counts})

    # 0 = inlier, 1 = outlier
    sns.barplot(data=pred_df, x='Label', y='Count')
    plt.xticks([0, 1], ['Inlier', 'Outlier'])
    plt.title('Prediction Distribution')
    plt.show()