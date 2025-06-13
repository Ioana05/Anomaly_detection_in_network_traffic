from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.deep_svdd import DeepSVDD
from sklearn.model_selection import RandomizedSearchCV # Import this
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, f1_score)
import numpy as np
from load_files import  load_and_preprocess_data,  change_proportion_of_data
from pyod.models.ocsvm import OCSVM

X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

# actually mi am dat seama ca nu cred ca nu este o idee buna sa folosesc one hot encoder pentru ca vor exista coloane cu o singura valoare de 1 swi restul 0,
# iar asta s ar putea sa influenteze resultatul pentru isolation forest

training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
# change proportion
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)

# After changing proportions:
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']   

X_train = X_train[y_train == 0]
y_train = y_train[y_train == 0]

# Generate folds
X_train_final, X_val, y_train_final, y_val = train_test_split(
 X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# def SVDD():
#     # 1. Generate balanced folds using the resampled data
#     folds = stratified_kfold_balance(X_train, y_train, n_splits=5) # Use the resampled data

#     n_epochs = 50
#     batch_size = 64

#     with tqdm(total=len(folds), desc="Training DeepSVDD") as pbar_folds:
#         for fold, (train_idx, test_idx) in enumerate(folds):
#             print(f"\nTraining Fold {fold + 1}")

#             # Get fold data from resampled data
#             X_train_fold = X_train[train_idx]
#             y_train_fold = y_train[train_idx]
#             X_val_fold = X_train[test_idx]
#             y_val_fold = y_train[test_idx]

#             # Train DeepSVDD model
#             n_features = X_train_fold.shape[1]
#             model = DeepSVDD(n_features=n_features, epochs=n_epochs, batch_size=batch_size, verbose=0)
#             model.fit(X_train_fold)

#             # Evaluate
#             predictions = model.predict(X_val_fold)
#             adjusted_predictions = np.where(predictions == 1, 0, 1)

#             f1 = f1_score(y_val_fold, adjusted_predictions)
#             acc = accuracy_score(y_val_fold, adjusted_predictions)

#             print(f"Fold {fold + 1} F1 Score: {f1:.4f}")
#             print(f"Fold {fold + 1} Accuracy score: {acc:.4f}")
#             pbar_folds.update(1)

#     # Count predictions (assuming this is done after the loop)
#     unique, counts = np.unique(predictions, return_counts=True)
#     pred_df = pd.DataFrame({'Label': unique, 'Count': counts})

#     # 0 = inlier, 1 = outlier
#     sns.barplot(data=pred_df, x='Label', y='Count')
#     plt.xticks([0, 1], ['Inlier', 'Outlier'])
#     plt.title('Prediction Distribution')
#     plt.show()

#     # Count predictions
#     unique, counts = np.unique(predictions, return_counts=True)
#     pred_df = pd.DataFrame({'Label': unique, 'Count': counts})

#     # 0 = inlier, 1 = outlier
#     sns.barplot(data=pred_df, x='Label', y='Count')
#     plt.xticks([0, 1], ['Inlier', 'Outlier'])
#     plt.title('Prediction Distribution')
#     plt.show()
def SVDD_with_validation():
    n_epochs = 70
    batch_size = 64

    print("Training DeepSVDD on training split...")

    n_features = X_train_final.shape[1]
    model = DeepSVDD(n_features=n_features, epochs=n_epochs, batch_size=batch_size, verbose=0)
    model.fit(X_train_final)

    # Optionally evaluate on validation set to check for overfitting
    val_preds = model.predict(X_val)
    val_preds_adj = np.where(val_preds == 1, 0, 1)
    print("Validation Results:")
    print(f"F1: {f1_score(y_val, val_preds_adj):.4f}")
    print(f"Accuracy: {accuracy_score(y_val, val_preds_adj):.4f}")

    # Final evaluation on test set
    test_preds = model.predict(X_test)
    test_preds_adj = np.where(test_preds == 1, 0, 1)
    print("Final Test Results:")
    print(f"F1: {f1_score(y_test, test_preds_adj):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, test_preds_adj):.4f}")

    # Plot prediction distribution
    unique, counts = np.unique(test_preds, return_counts=True)
    pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
    sns.barplot(data=pred_df, x='Label', y='Count')
    plt.xticks([0, 1], ['Inlier', 'Outlier'])
    plt.title('Prediction Distribution')
    plt.show()


def OCSVM_with_validation(X_train_final, X_val, y_val, X_test, y_test):
    kernels = ['rbf']
    for ker in kernels:
        print(f"Training OCSVM on training split with gamma = 1.0, nu = 0.3 ...")

        # Initialize and train the OCSVM model
        model = OCSVM(nu=0.3, kernel=ker, gamma= 1.0)
        model.fit(X_train_final)

        # Validation predictions
        val_preds = model.predict(X_val)
        # OCSVM predicts 0 for inliers and 1 for outliers
        # Adjust labels if your y_val labels are 0 (normal) and 1 (anomaly)
        # Here we assume 0 = normal, 1 = anomaly, so no inversion needed
        print("Validation Results:")
        print(f"F1: {f1_score(y_val, val_preds):.4f}")
        print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")

        # Test predictions
        test_preds = model.predict(X_test)
        print("Final Test Results:")
        print(f"Classification Report: {classification_report(y_test, test_preds)}")

        # Plot prediction distribution on test set
        # unique, counts = np.unique(test_preds, return_counts=True)
        # pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
        # sns.barplot(data=pred_df, x='Label', y='Count')
        # plt.xticks([0, 1], ['Inlier', 'Outlier'])
        # plt.title('Prediction Distribution')
        # plt.show()
print("Results for OCSVM:")
OCSVM_with_validation(X_train_final, X_val, y_val, X_test, y_test)


print("Results for SVDD:")
# SVDD_with_validation()

