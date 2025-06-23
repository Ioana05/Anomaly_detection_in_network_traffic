from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from .KNN_class import kNNAnomalyDetector
from .IF_class import IsolationForestScikit
from .SVDD_class import OCSVMClassifier
from .AE_class import AutoEncoder
from ...pipeline import  load_and_preprocess_data,  change_proportion_of_data
from .visualisations import accuracies_on_attacks, evaluate, ROC, plot_predictions_distribution
import pandas as pd
import numpy as np

X_train, y_train, X_test, y_test, training_set_df, testing_set_df = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

# recreem dataframe uri din datele de antrenare si testare
training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train
testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

# adaugam inapoi coloana attack_cat ca sa putem verifica mai tarziu care a fost acuratetea pentru fiecare atac
testing_set_resampled['attack_cat'] = testing_set_df['attack_cat']
print(f"Dimensiunea setului de test {len(testing_set_resampled)}")

train_anomalies = 0.1
# Schimbam rata de infectare a setului de antrenare
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)

# reseparam datele de label uri
y_train = training_set_resampled['label']                 
X_train = training_set_resampled.drop(columns=['label'])  

y_test = testing_set_resampled['label']
X_test = testing_set_resampled.drop(columns = ['label', 'attack_cat'])

# Initializam modelele
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
    "SVDD": svdd_model
}
attacks_accuracies = {}
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


        # Adaugă predicțiile în dataframe-ul original (pentru analiză pe tip de atac)
        temp_df = testing_set_resampled.copy()
        temp_df['predicted'] = y_pred

        print(f"Acuratețe pe fiecare tip de atac pentru {name}:")
        tip_atac = temp_df['attack_cat'].unique()
        if name not in attacks_accuracies:
            attacks_accuracies[name] = []
        for attack in tip_atac:
            subset = temp_df[temp_df['attack_cat'] == attack]
            acc = accuracy_score(subset['label'], subset['predicted'])
            attacks_accuracies[name].append(acc)
            print(f"  → {attack}: {acc:.2f}")

            
    except Exception as e:
        print(f"Error fitting or evaluating {name}: {e}")

attacks_accuracies['Tipuri date'] = tip_atac

# Evaluam
print("\n\ Soft Voting Classifier Performance ")

y_pred_soft_voting = voting_soft_method.predict(X_test)

# PT PLOTARE

# Adaugă predicțiile în dataframe-ul original (pentru analiză pe tip de atac)
temp_df = testing_set_resampled.copy()
temp_df['predicted'] = y_pred_soft_voting

print(f"Acuratețe pe fiecare tip de atac pentru Soft Voting:")
tip_atac = temp_df['attack_cat'].unique()
attacks_accuracies['Soft Voting'] = []
for attack in tip_atac:
    subset = temp_df[temp_df['attack_cat'] == attack]
    acc = accuracy_score(subset['label'], subset['predicted'])
    attacks_accuracies['Soft Voting'].append(acc)
    print(f"  → {attack}: {acc:.2f}")


# PLOTAM DISTRIBUTIILE ATACURILOR
accuracies_on_attacks(attacks_accuracies)

# EVALUAM SOFT VOTING
y_proba_soft_voting = voting_soft_method.predict_proba(X_test)[:, 1] 
evaluate(y_test, y_pred_soft_voting, y_proba_soft_voting)

# MAX PROBABILITY
print("\n\n Max Probability Voting Ensemble Performance ")

# metoda Max probability presupune sa luam doar scorul maxim pentru fiecare punct ca acesta sa fie anomalie
y_proba_max_ensemble = np.max([m.predict_proba(X_test)[:, 1] for m in voting_soft_method.estimators_], axis=0)

# aplicam un threshold ales de noi 
max_proba_threshold = 0.5
y_pred_max_ensemble = (y_proba_max_ensemble > max_proba_threshold).astype(int)
print(f"Using Threshold: {max_proba_threshold} for 'Max Probability' predictions ")
evaluate(y_test, y_pred_max_ensemble, y_proba_max_ensemble)

# Vizualizari ROC curve pentru metoda soft voting si probability ensemble
ROC(y_test, y_proba_soft_voting, y_proba_max_ensemble)


# Plotam cate anomalii au prezis cele doua abordari
plot_predictions_distribution(y_pred_soft_voting, y_pred_max_ensemble)