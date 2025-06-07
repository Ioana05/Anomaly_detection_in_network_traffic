from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.exceptions import NotFittedError
import warnings

class kNNAnomalyDetector(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, batch_size=1000, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.contamination = contamination
        self._estimator_type = "classifier" # Correctly set for sklearn compatibility
    
    def fit(self, X, y=None):
        # Input validation
        if y is None:
            X = check_array(X)
            # If no labels are provided, assume all are normal for fitting unsupervised
            y = np.zeros(X.shape[0], dtype=int) 
            self.classes_ = np.array([0, 1]) # Still classify into 0 and 1
        else:
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
        
        self.n_features_in_ = X.shape[1]
        
        # Handle single-class case (e.g., if all training data is one class)
        if len(self.classes_) == 1:
            warnings.warn(
                f"Only one class ({self.classes_[0]}) present in training data. "
                "This might indicate a problem with your data. "
                "The model will predict this single class for all inputs."
            )
            self.single_class_ = self.classes_[0]
            self.is_fitted_ = True # Mark as fitted even if trivial
            return self
        
        # Use only normal samples (label 0) for training if available
        # This is crucial for unsupervised anomaly detection where 0 is 'normal'
        if 0 in self.classes_ and np.sum(y == 0) > 0:
            X_train_normal = X[y == 0]
        else:
            warnings.warn(
                "No normal samples (label=0) found in training data, "
                "or all training data is of a single non-zero class. "
                "The kNNAnomalyDetector will train on all provided samples. "
                "Ensure your training data correctly represents 'normal' (label 0)."
            )
            X_train_normal = X # Fallback to using all data if no '0' labels or all are '0'
        
        # Fit nearest neighbors on normal data
        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_train_normal)
        
        # Calculate threshold for anomaly detection
        distances, _ = self.knn_.kneighbors(X_train_normal)
        self.threshold_ = np.percentile(distances[:, -1], 100 * (1 - self.contamination))
        
        self.is_fitted_ = True # Mark the estimator as fitted
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_') # Use the new attribute
        X = check_array(X)
        
        if hasattr(self, 'single_class_'): # If trained on single class
            return np.full(X.shape[0], self.single_class_, dtype=int)
        
        scores = self._calculate_anomaly_scores(X)
        return np.where(scores > self.threshold_, 1, 0) # 1 for anomaly, 0 for normal
    
    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_') # Use the new attribute
        X = check_array(X)
        
        if hasattr(self, 'single_class_'): # If trained on single class
            proba = np.zeros((X.shape[0], len(self.classes_)))
            # If single_class_ is 0, all prob to col 0. If 1, all prob to col 1.
            proba[:, np.where(self.classes_ == self.single_class_)[0][0]] = 1.0
            return proba
        
        scores = self._calculate_anomaly_scores(X)
        
        # Normalize scores to probabilities
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            # If all scores are the same, assign 0.5 probability (or based on threshold)
            # A more robust handling for flat scores might be needed depending on context.
            # For now, treat all as equally likely or based on initial prediction threshold
            proba_anomaly = (scores > self.threshold_).astype(float) 
        else:
            proba_anomaly = (scores - min_score) / (max_score - min_score + 1e-10) # Add epsilon to avoid division by zero
        
        # Return probabilities for both classes [P(normal), P(anomaly)]
        return np.column_stack([1 - proba_anomaly, proba_anomaly])
    
    def _calculate_anomaly_scores(self, X):
        # Ensure knn_ is fitted before calculating scores
        if not hasattr(self, 'knn_') or self.knn_ is None:
            raise NotFittedError("The kNN model has not been fitted yet.")

        scores = np.zeros(X.shape[0])
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            # Use self.knn_ directly, it's already fitted
            distances, _ = self.knn_.kneighbors(X[start:end])
            scores[start:end] = distances[:, -1] # k-th distance as anomaly score
        return scores
    
    def score(self, X, y):
        # Default score uses accuracy for classifiers
        return accuracy_score(y, self.predict(X))