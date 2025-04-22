import random
from sklearn.ensemble import IsolationForest
from load_files import training_set, testing_set
from sklearn.metrics import accuracy_score, classification_report
from load_files import final_testing_set, final_training_set

# ==========================================
# 4. Modelul Isolation Forest
# ==========================================
anomaly_ratio = final_training_set.sum() / len(testing_set)

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.5,  # ajustează în funcție de distribuția anomaliilor tale
    random_state=42
)

# Fit pe datele de antrenament
iso_forest.fit(final_training_set)

# Predict pe datele de test
preds = iso_forest.predict(testing_set)

# Convertim -1 = anomalii → 1, 1 = normal → 0
preds = [1 if x == -1 else 0 for x in preds]

# ==========================================
# 5. Rezultate
# ==========================================
print("Accuracy:", accuracy_score(testing_set['label'], preds))
print(classification_report(testing_set['label'], preds))
