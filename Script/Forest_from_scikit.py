import random
from sklearn.ensemble import IsolationForest
from load_files import training_set, testing_set
from sklearn.metrics import accuracy_score, classification_report
from load_files import final_testing_set, final_training_set


# Drop labels and ID from the training/testing data
X_train = final_training_set.drop(columns=['label', 'id'])
X_test = final_testing_set.drop(columns=['label', 'id'])
y_test = final_testing_set['label']

# Train the Isolation Forest
iso_forest = IsolationForest(n_estimators=60, max_samples=200,  random_state=42)
iso_forest.fit(X_train)

# Predict anomalies
preds = iso_forest.predict(X_test)
preds = [1 if x == -1 else 0 for x in preds]  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

# Evaluate
print("Accuracy:", accuracy_score(y_test, preds))
# print(classification_report(y_test, preds))
