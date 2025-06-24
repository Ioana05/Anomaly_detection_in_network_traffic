from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.deep_svdd import DeepSVDD
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix, roc_auc_score, f1_score)
import numpy as np
from Script.pipeline import  load_and_preprocess_data,  change_proportion_of_data
from pyod.models.ocsvm import OCSVM

X_train, y_train, X_test, y_test,  _, _ = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
# schimbam rata de infectare a setului de antrenare
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)

# Dupa ce am schimbat rata, separam iarasi datele de antrenare de datele de test
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']   

X_train = X_train[y_train == 0]
y_train = y_train[y_train == 0]

# separam o parte din date pentru validare
X_train_final, X_val, y_train_final, y_val = train_test_split(
 X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

def plot_prediction_distribution(test_preds):
    unique, counts = np.unique(test_preds, return_counts=True)
    pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
    sns.barplot(data=pred_df, x='Label', y='Count')
    plt.xticks([0, 1], ['Inlier', 'Outlier'])
    plt.title('Prediction Distribution')
    plt.show()

# am incercat inainte si SVDD folosind stratified kfold(nu a imbunatatit scorurile)

# DeepSvdd din pyod nu a avut rezultate foarte bune
def SVDD_with_validation():
    n_epochs = 70
    batch_size = 64
    n_features = X_train_final.shape[1]

    #initializam si antrenam modelul
    model = DeepSVDD(n_features=n_features, epochs=n_epochs, batch_size=batch_size, verbose=0)
    model.fit(X_train_final)

    # evaluam si pe validation set ca sa reducem sansele de overfitting
    val_preds = model.predict(X_val)
    
    print("Validation Results:")
    print(f"F1: {f1_score(y_val, val_preds):.4f}")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")

    # Evaluare finala pe setul de test
    test_preds = model.predict(X_test)
    print("Final Test Results:")
    print(f"F1: {f1_score(y_test, test_preds):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")

    # Plot prediction distribution
    unique, counts = np.unique(test_preds, return_counts=True)
    pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
    sns.barplot(data=pred_df, x='Label', y='Count')
    plt.xticks([0, 1], ['Inlier', 'Outlier'])
    plt.title('Prediction Distribution')
    plt.show()


def OCSVM_with_validation(X_train_final, X_val, y_val, X_test, y_test):

    # Initializam si antrenam modelul OCSVM
    # utilizam kernelul rbf deoarece vrem ca OCSVM sa aiba comportament asemanator cu DeepSVDD
    used_nu = 0.2 # parametrul obtinut in urma experimentelor
    used_gamma = 'scale'
    model = OCSVM(nu=used_nu, kernel='rbf', gamma=used_gamma)
    print(f"Training OCSVM on training split with gamma = {used_gamma}, nu = {used_nu} ...")
    model.fit(X_train_final)

    # Calculam predictiile pe datele de validare
    val_preds = model.predict(X_val)
  
    print("Validation Results:")
    print(f"F1: {f1_score(y_val, val_preds):.4f}")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")

    # Calculam predictiile pe setul de test
    test_preds = model.predict(X_test)
    print("Final Test Results:")
    print(f"Classification Report: {classification_report(y_test, test_preds)}")

    # Plot prediction distribution on test set
    # plot_prediction_distribution(test_preds=test_preds)

print("Results for OCSVM:")
OCSVM_with_validation(X_train_final, X_val, y_val, X_test, y_test)


# print("Results for SVDD:")
# SVDD_with_validation()

