from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from load_files import load_and_preprocess_data
from KNN_Scikit_class import kNNAnomalyDetector
from Scikit_Iso_class import IsolationForestScikit
from SVDD_class import OCSVMClassifier
from Autoencoder_class import AutoEncoder
import matplotlib.pyplot as plt
from load_files import  load_and_preprocess_data,  change_proportion_of_data
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
import random
import pandas as pd
import numpy as np


X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

# actually mi am dat seama ca nu cred ca nu este o idee buna sa folosesc one hot encoder pentru ca vor exista coloane cu o singura valoare de 1 swi restul 0,
# iar asta s ar putea sa influenteze resultatul pentru isolation forest

training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

train_anomalies = 0.1
# change proportion
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)

# After changing proportions:
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 


knn_model = kNNAnomalyDetector()
isolation_forest_model = IsolationForestScikit()
svdd_model = OCSVMClassifier()
autoEncoder_model = AutoEncoder()


voting_soft_method = VotingClassifier(
    estimators=[
        ('knn', kNNAnomalyDetector()), 
        ('iso_forest', IsolationForestScikit()),
        ('autoencoder', AutoEncoder()),
        ('ocsvm', OCSVMClassifier())
    ],
    voting='soft',
    weights=[1,1,2,1] 
)

# Fit metoda de voting 

voting_soft_method.fit(X_train, y_train)
print("Soft Voting Classifier Fitted.")

# Evaluam individual modelele
models = {
    "kNN": knn_model,
    "Isolation Forest": isolation_forest_model,
    "Autoencoder": autoEncoder_model,
    "OCSVM": svdd_model
}

for name, model in models.items():
    print(f"\n--- {name} ---")
    try:
       
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] > 1: 
                y_proba_anomaly = y_proba[:, 1]
                print("ROC AUC:", roc_auc_score(y_test, y_proba_anomaly))
            else:
                print("ROC AUC: Not available (predict_proba does not return probabilities for both classes).")
        else:
            print("ROC AUC: Not available (model does not have predict_proba).")
            
    except Exception as e:
        print(f"Error fitting or evaluating {name}: {e}")

# Evaluam
print("\n\ Soft Voting Classifier Performance ")

y_pred_soft_voting = voting_soft_method.predict(X_test)
y_proba_soft_voting = voting_soft_method.predict_proba(X_test)[:, 1] 

print(classification_report(y_test, y_pred_soft_voting))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_soft_voting))
print("ROC AUC:", roc_auc_score(y_test, y_proba_soft_voting))


print("\n\n Max Probability Voting Ensemble Performance ")

# Calculate y_proba_max using the fitted base estimators from the soft voting classifier
# This takes the element-wise maximum probability for the anomaly class (1) across all base models
y_proba_max_ensemble = np.max([m.predict_proba(X_test)[:, 1] for m in voting_soft_method.estimators_], axis=0)

# aplicam un threshold ales de noi 
max_proba_threshold = 0.5
y_pred_max_ensemble = (y_proba_max_ensemble > max_proba_threshold).astype(int)

print(f"Using Threshold: {max_proba_threshold} for 'Max Probability' predictions ")
print(classification_report(y_test, y_pred_max_ensemble))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_max_ensemble))
print("ROC AUC (Max Probability Ensemble):", roc_auc_score(y_test, y_proba_max_ensemble))

# Vizualizari pt ambele
plt.figure(figsize=(12, 6)) 

# ROC Curve pt metoda de Soft Voting
plt.subplot(1, 2, 1)
fpr_soft, tpr_soft, _ = roc_curve(y_test, y_proba_soft_voting)
plt.plot(fpr_soft, tpr_soft, color='darkorange', lw=2, 
         label=f'Soft Voting (AUC = {roc_auc_score(y_test, y_proba_soft_voting):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Soft Voting ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)

# ROC Curve for pt Probability Ensemble
plt.subplot(1, 2, 2)
fpr_max, tpr_max, _ = roc_curve(y_test, y_proba_max_ensemble)
plt.plot(fpr_max, tpr_max, color='green', lw=2, 
         label=f'Max Proba Ensemble (AUC = {roc_auc_score(y_test, y_proba_max_ensemble):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Max Probability Ensemble ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()

# You might also want to plot prediction distributions for these two ensembles
# For example, to see how many anomalies each method predicts at the chosen threshold.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
pd.Series(y_pred_soft_voting).value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Normal (0)', 'Anomaly (1)'], rotation=0)
plt.title('Soft Voting Prediction Distribution')
plt.ylabel('Number of Samples')
plt.xlabel('Predicted Class')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
pd.Series(y_pred_max_ensemble).value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.xticks([0, 1], ['Normal (0)', 'Anomaly (1)'], rotation=0)
plt.title('Max Probability Ensemble Prediction Distribution')
plt.ylabel('Number of Samples')
plt.xlabel('Predicted Class')
plt.grid(axis='y')

plt.tight_layout()
plt.show()