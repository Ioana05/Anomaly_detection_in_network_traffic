from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.deep_svdd import DeepSVDD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV # Import this
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, f1_score)
import numpy as np
from load_files import  load_and_preprocess_data,  change_proportion_of_data
from pyod.models.ocsvm import OCSVM


class OCSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel = 'rbf', nu = 0.1, gamma = 0.3):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model_ = None

    def fit(self, X, y=None):  
        self.model_ = OCSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model_.fit(X)
        return self
    
    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
        
        # functia de decizie returneaza scoruri mai mari pt punctele cu posibilitate mai mare sa fie anomalii
        decision_scores = self.model_.decision_function(X)
        
        # Normalizam scorurile incat sa fie in intervalul [0,1]
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        if max_score == min_score:  # handle division by zero
            proba = np.zeros_like(decision_scores)
        else:
            proba = (decision_scores - min_score) / (max_score - min_score)
    
        # Le facem in Scikit format
        return np.vstack([1 - proba, proba]).T
    
    def plot_prediction_distribution(self, X):
        preds = self.predict(X)
        unique, counts = np.unique(preds, return_counts=True)
        pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
        sns.barplot(data=pred_df, x='Label', y='Count')
        plt.xticks([0, 1], ['Inlier', 'Outlier'])
        plt.title('PyOD OCSVM Prediction Distribution')
        plt.show()