import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# PLOTAM DISTRIBUTIILE ATACURILOR

def accuracies_on_attacks(attacks_accuracies):
    df = pd.DataFrame(attacks_accuracies)

    # transformăm DataFrame-ul din "wide" format în "long"  pt Seaborn
    df_melted = df.melt(id_vars=['Tipuri date'], var_name='Model', value_name='Acuratete')

    plt.figure(figsize=(14, 7)) # Ajustează dimensiunea figurii
    sns.barplot(x='Tipuri date', y='Acuratete', hue='Model', data=df_melted, palette='viridis')

    plt.title('Acuratețea Fiecărui Model pe Tip de Atac', fontsize=16)
    plt.xlabel('Tipuri date', fontsize=12)
    plt.ylabel('Acuratețe', fontsize=12)
    plt.ylim(0, 1.05) # Acuratețea este între 0 și 1
    plt.xticks(rotation=45, ha='right', fontsize=10) # Roteste etichetele axei X pentru lizibilitate
    plt.yticks(fontsize=10)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left') # Plaseaza legenda in afara graficului
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Adauga un grid pe axa Y
    plt.tight_layout() # Ajustează layout-ul pentru a evita suprapunerile
    plt.show()

def evaluate(y_test, y_pred, y_proba):
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))


def ROC(y_test, y_proba_soft_voting, y_proba_max_ensemble):
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

    plt.tight_layout() 
    plt.show()

def plot_predictions_distribution(y_pred_soft_voting, y_pred_max_ensemble):
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