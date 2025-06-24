import random
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
from Script.pipeline import change_proportion_of_data, load_and_preprocess_data
import matplotlib.pyplot as plt
X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )


training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
# Schimbam rata de infectare doar pe setul de antrenare
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)


# Dupa ce am schimbat rata, separam iarasi datele de antrenare de datele de test
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 
     
if 1 in y_train.unique():
    contamination_rate = y_train.value_counts(normalize=True)[1]
else:
    contamination_rate = 0.0

# valori obtinute in urma experimentelor
trees = 100
max_samples = 256
random_state = 42

# construim ansamblul de arbori

iso_forest = IsolationForest(n_estimators=trees, max_samples=max_samples,  random_state=random_state,  contamination=0.5, max_features=0.7, bootstrap=False)
iso_forest.fit(X_train)

# Predict anomalies
preds = iso_forest.predict(X_test)
preds = [1 if x == -1 else 0 for x in preds]  # Convertim -1 în 1 (pentru anomalii), 1 în 0 (puncte normale)

# Evaluăm predicțiile făcute
print(f"Proportion of anomalies in train dataset: {train_anomalies}")
print(f"Number of trees {trees}, max_samples {max_samples}, random_state {random_state}")
print("Accuracy:", accuracy_score(y_test, preds), "Recall:", recall_score(y_test, preds), "Precision: ", precision_score(y_test, preds), "F1-score:", f1_score(y_test, preds))
print(classification_report(y_test, preds))


#  avem nevoie de scoruri de data asta, nu labe luri
# scorurile mai mici implica anomalii
anomaly_scores = iso_forest.decision_function(X_test)

#  pt ROC de obicei , scorurile mai mari indica clasa pozitiva, dar 
# pt ca IsoForest returneaza scoruri negative pt anomalii, inversam
positive_scores = 1 - anomaly_scores
fpr, tpr, thresholds = roc_curve(y_test, positive_scores, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
