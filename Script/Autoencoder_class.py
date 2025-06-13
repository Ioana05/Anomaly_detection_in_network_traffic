from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from load_files import load_and_preprocess_data
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from sklearn.base import BaseEstimator, ClassifierMixin

class AutoEncoder(ClassifierMixin, BaseEstimator):

    def __init__(self, threshold_percentile = 70, epochs = 50, batch_size = 64):
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.error_threshold = None  #nu sunt sigura daca trenbuie si asta
        self.model = None
        self._estimator_type = "classifier"


    def fit(self, X, y = None):
        # nu uita sa trimiti doar date normale(tu l ai antrenat semi supervised, dar poti incerca si unsupervised)
        input_dim = X.shape[1]
        
        # Build model
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        bottleneck = Dense(8, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(bottleneck)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        output_layer = Dense(input_dim, activation='linear')(decoded)


        self.model_ = Model(inputs=input_layer, outputs=output_layer)
        self.model_.compile(optimizer=Adam(0.001), loss=Huber(delta=1.0))

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        X_normal = X[y == 0] if y is not None else X

        self.model_.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop]
        )

        reconstructions = self.model_.predict(X_normal, verbose=0)
        reconstruction_error = np.mean(np.square(X_normal - reconstructions), axis=1)
        self.error_threshold_ = np.percentile(reconstruction_error, self.threshold_percentile)

        return self
    
    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        reconstructions = self.model_.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - reconstructions), axis=1)
        return (reconstruction_error > self.error_threshold_).astype(int)
    

    def predict_proba(self, X):
        # For soft voting 
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
            
        reconstructions = self.model_.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - reconstructions), axis=1)
        
        # Convertim erorile in scoruri intre 0 si 1 
        proba = np.clip(reconstruction_error / (2 * self.error_threshold_), 0, 1)
        #  transformamm scrurile de anomalii inintr o matrice cu doua coloane, asa cum se asteapta scikit
        #  sa primeasca de la functia predict_proba
        return np.vstack([1 - proba, proba]).T  # [prob_class_0, prob_class_1]