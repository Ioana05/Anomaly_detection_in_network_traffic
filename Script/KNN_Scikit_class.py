import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class kNNAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=2, batch_size=1000, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.contamination = contamination
        self.threshold_ = None
        self.knn_ = None
    
    def fit(self, X, y=None):

        # Foolosim doar labelurile normale pentru antrenare
        if y is not None:
            X = X[y == 0]
            
        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn_.fit(X)
        
        #  threshold
        distances, _ = self.knn_.kneighbors(X)
        kth_distances = distances[:, -1]
        self.threshold_ = np.percentile(kth_distances, 100 * (1 - self.contamination))
        
        return self
    
    def predict(self, X):
        if self.knn_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        scores = self._calculate_anomaly_scores(X)
        return (scores > self.threshold_).astype(int)
    
    def predict_proba(self, X):
        if self.knn_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        scores = self._calculate_anomaly_scores(X)
        
        # Normalizam si aici scorurile
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            proba = np.zeros_like(scores)
        else:
            proba = (scores - min_score) / (max_score - min_score)
        
        # Return probabilities for both classes
        return np.vstack([1 - proba, proba]).T
    
    def _calculate_anomaly_scores(self, X):
        #  calculam scorurile de anomalie
        scores = np.zeros(len(X))
        
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch = X[start:end]
            
            distances, _ = self.knn_.kneighbors(batch)
            scores[start:end] = distances[:, -1]  
            
        return scores
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))