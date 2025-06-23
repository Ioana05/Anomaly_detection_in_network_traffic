import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from pyod.models.ocsvm import OCSVM


class OCSVMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, kernel = 'rbf', nu = 0.2, gamma = 'scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model_ = None
        self._estimator_type = "classifier"

    def fit(self, X, y=None): 
        # filtram doar datele normale, vrem sa antrenam modelul doar cu ele
        X_train_normal = X[y == 0] 
        self.model_ = OCSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model_.fit(X_train_normal)
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
        if max_score == min_score: 
            proba = np.zeros_like(decision_scores) # daca toate punctele au acelasi scor, consideram ca ele formeaza un 'cluster' => OCSVM considera datele normale
        else:
            proba = (decision_scores - min_score) / (max_score - min_score) 
    
        # Le facem in Scikit format
        return np.column_stack([1 - proba, proba])
    
    def plot_prediction_distribution(self, X):
        preds = self.predict(X)
        unique, counts = np.unique(preds, return_counts=True)
        pred_df = pd.DataFrame({'Label': unique, 'Count': counts})
        sns.barplot(data=pred_df, x='Label', y='Count')
        plt.xticks([0, 1], ['Inlier', 'Outlier'])
        plt.title('PyOD OCSVM Prediction Distribution')
        plt.show()