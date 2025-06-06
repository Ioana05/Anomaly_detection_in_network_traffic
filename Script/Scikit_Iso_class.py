import random
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve, auc
from load_files import change_proportion_of_data, load_and_preprocess_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class IsolationForestScikit(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators = 100, max_samples = 256, random_state = 42, contamination = 'auto', max_features = 0.7, bootstrap = False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.model_ = None
    
    def fit(self, X, y=None):  
        if self.contamination == 'auto' and y is not None:
            self.contamination = np.mean(y == 1)
        
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap
        )
        self.model_.fit(X)
        return self
    

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
        # Convertim -1 la 1, si 1 la 0
        return np.where(self.model_.predict(X) == -1, 1, 0)
    

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        # aici scorurile mai mici sunt considerate cu sanse mai mari sa fie anomalii
        scores = self.model_.decision_function(X)
        
        #  normalizam si inversam ca sa fie asa cum ne am astepta
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:  
            proba = np.zeros_like(scores)
        else:
            proba = 1 - ((scores - min_score) / (max_score - min_score))
        
        # Il facem in formatul scikit
        return np.vstack([1 - proba, proba]).T
    
    def plot_roc_curve(self, X, y_true):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        # Get anomaly probabilities for class 1
        proba = self.predict_proba(X)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, proba, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Isolation Forest ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()