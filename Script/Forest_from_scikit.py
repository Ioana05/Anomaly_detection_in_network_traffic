import random
import pandas as pd
from sklearn.ensemble import IsolationForest
from load_files import training_set, testing_set
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve, auc
from load_files import X_test, y_test, X_train, y_train, change_proportion_of_data
import matplotlib.pyplot as plt


training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
test_anomalies = 0.1
# change proportion
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)
testing_set = change_proportion_of_data(testing_set_resampled, percentage_anomalies=test_anomalies, total=30000)

# After changing proportions:
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 

X_test = testing_set.drop(columns=['label'])              
y_test = testing_set['label']           

if 1 in y_train.unique():
    contamination_rate = y_train.value_counts(normalize=True)[1]
else:
    contamination_rate = 0.0

print(contamination_rate)

trees = [100]
max_samples = [256]
random_state = [42]
# Train the Isolation Forest
for tree in trees:
    for sample in max_samples:
        for state in random_state:
            iso_forest = IsolationForest(n_estimators=tree, max_samples=sample,  random_state=state,  contamination=contamination_rate, max_features=0.7, bootstrap=False)
            iso_forest.fit(X_train)

            # Predict anomalies
            preds = iso_forest.predict(X_test)
            preds = [1 if x == -1 else 0 for x in preds]  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

            # Evaluate
            # print("Made contamination 0.03")
            print(f"Proportion of anomalies in train dataset: {train_anomalies} and in test dataset: {test_anomalies}")
            print(f"Number of trees {tree}, max_samples {sample}, random_state {state}")
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
