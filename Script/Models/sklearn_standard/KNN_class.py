from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_X_y
from sklearn.metrics import accuracy_score
import warnings

class kNNAnomalyDetector(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=8, batch_size=1000, contamination=0.10):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.contamination = contamination
        self._estimator_type = "classifier"
    
    def fit(self, X, y=None):
        # Validam inputul
        if y is None:
            X = check_array(X)
            # Daca nu avem label-uri, plecam de la premisa ca toate sunt normale
            y = np.zeros(X.shape[0], dtype=int) 
            self.classes_ = np.array([0, 1]) # Vor exista doua label uri posibile
        else:
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
        
        self.n_features_in_ = X.shape[1]
        
        # filtram training set ul incat sa pastram doar punctele normale(vrem sa facem antrenarea doar pe ele)
        if 0 in self.classes_ and np.sum(y == 0) > 0:
            X_train_normal = X[y == 0]
        else:
            warnings.warn(
                "No normal samples (label=0) found in training data, "
            )
            X_train_normal = X  #pentru cazul in care nu exista puncte normale sau toate punctele sunt normale
        
        # antrenam modelul pe datele filtrate anterior
        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_train_normal)
        
        # calculam threshold ul folosind contaminarea data ca parametru
        distances, _ = self.knn_.kneighbors(X_train_normal)
        self.threshold_ = np.percentile(distances[:, -1], 100 * (1 - self.contamination))
        
        self.is_fitted_ = True 
        return self
    
    def predict(self, X):
        if self.is_fitted_ != True: 
            raise RuntimeError("The model must be fitted before prediction")
        X = check_array(X)
        
        scores = self._calculate_anomaly_scores(X)
        return np.where(scores > self.threshold_, 1, 0) # 1 pentru anomalii, 0 pentru date normale
    
    def predict_proba(self, X):
        if self.is_fitted_ != True: 
            raise RuntimeError("The model must be fitted before prediction")
        X = check_array(X)
        
        scores = self._calculate_anomaly_scores(X)
        
        # Normalizam scorurile
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            # daca toate scorurile sunt egale,  proba_anomaly va fi 0 pentru toate. In acest caz, folosim un threshold pentru a cataloga punctele care se afla la o distanta mai mare decat threshold ul stabilit, ca fiind anomalii
            proba_anomaly = (scores > self.threshold_).astype(float) 
        else:
            proba_anomaly = (scores - min_score) / (max_score - min_score + 1e-10) # Adaugam epsilon pentru cazurile in care min_score si max_score au valori foarte apropiate
        
        # Returnam probabilitatile pentru ambele clase [P(normal), P(anomaly)]
        return np.column_stack([1 - proba_anomaly, proba_anomaly])
    
    def _calculate_anomaly_scores(self, X):
        if not hasattr(self, 'knn_') or self.knn_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        scores = np.zeros(X.shape[0])
        # calculam scorurile in batch-uri, altfel ajungem la OOM error
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            distances, _ = self.knn_.kneighbors(X[start:end]) # distances va fi o matrice de forma (batch_size, k)
            scores[start:end] = distances[:, -1] # folosim a k-a distanta ca scor de anomalie
        return scores
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))