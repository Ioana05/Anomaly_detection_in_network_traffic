import random
from sklearn.ensemble import IsolationForest
from load_files import training_set, testing_set
from sklearn.metrics import accuracy_score, classification_report
from load_files import X_test, y_test, X_res, y_res, X_train, y_train

contamination = y_res.mean()  # Typically ~0.1-0.2 for UNSW-NB15
print(contamination)

trees = [100]
max_samples = [256]
random_state = [42]
# Train the Isolation Forest
for tree in trees:
    for sample in max_samples:
        for state in random_state:
            iso_forest = IsolationForest(n_estimators=tree, max_samples=sample,  random_state=state,  contamination=contamination, max_features=0.7, bootstrap=False)
            iso_forest.fit(X_res)

            # Predict anomalies
            preds = iso_forest.predict(X_test)
            preds = [1 if x == -1 else 0 for x in preds]  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

            # Evaluate
            print(f"Number of trees {tree}, max_samples {sample}, random_state {state}")
            print("Accuracy:", accuracy_score(y_test, preds))
            print(classification_report(y_test, preds))


