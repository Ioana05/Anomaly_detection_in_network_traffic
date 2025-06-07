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


# X_train = X_train[y_train == 0]
# y_train = y_train[y_train == 0]

knn_model = kNNAnomalyDetector()
isolation_forest_model = IsolationForestScikit()
# print(check_estimator(isolation_forest_model))
svdd_model = OCSVMClassifier()
autoEncoder_model = AutoEncoder()


voting_method = VotingClassifier(
    estimators=[
        ('knn', kNNAnomalyDetector()),
         ('iso_forest', IsolationForestScikit()),
         ('autoencoder', AutoEncoder() ),
         ('ocsvm', OCSVMClassifier())
    ],
    voting = 'hard'
)

voting_method.fit(X_train, y_train)

# evaluam si separat modelele
models = {
    "kNN": kNNAnomalyDetector(),
    "Isolation Forest": IsolationForestScikit(),
    "Autoencoder": AutoEncoder(),
    "OCSVM": OCSVMClassifier()
}


print("Individual Model Performance:")
for name, model in models.items():
    model.fit(X_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# 2. Evaluate the Voting Classifier
print("\n\nVoting Classifier Performance:")

# Fit the voting classifier (already done in your code)
# voting_clf.fit(X_train, y_train)

# Get predictions and probabilities
y_pred_voting = voting_method.predict(X_test)
y_proba_voting = voting_method.predict_proba(X_test)[:, 1]  # Probability of being anomaly

# Print metrics
print(classification_report(y_test, y_pred_voting))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))
print("ROC AUC:", roc_auc_score(y_test, y_proba_voting))

# 3. Visualization
plt.figure(figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_voting)
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'Voting Classifier (AUC = {roc_auc_score(y_test, y_proba_voting):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Prediction Distribution
plt.subplot(1, 2, 2)
pd.Series(y_pred_voting).value_counts().plot(kind='bar')
plt.xticks([0, 1], ['Normal', 'Anomaly'], rotation=0)
plt.title('Prediction Distribution')
plt.show()