from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from load_files import load_and_preprocess_data
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber


X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

X_train = X_train[y_train == 0]
y_train = y_train[y_train == 0]

def Autoencoder(specified_threshold):

    input_dim = X_train.shape[1]  # după SMOTE/feature selection

    # Arhitectura autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    # encoded = LeakyReLU(alpha=0.01)(encoded) 
    encoded = Dense(64, activation='relu')(encoded)
    # encoded = LeakyReLU(alpha=0.01)(encoded) 
    encoded = Dense(32, activation='relu')(encoded)
    # encoded = LeakyReLU(alpha=0.01)(encoded) 
    bottleneck = Dense(8, activation='relu')(encoded)
    # bottleneck = LeakyReLU(alpha=0.01)(bottleneck) 
    decoded = Dense(32, activation='relu')(bottleneck)
    # decoded = LeakyReLU(alpha=0.01)(decoded) 
    decoded = Dense(64, activation='relu')(decoded)
    # decoded = LeakyReLU(alpha=0.01)(decoded)
    
    decoded = Dense(128, activation='relu')(decoded)
    # decoded = LeakyReLU(alpha=0.01)(decoded) 
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    # autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    autoencoder.compile(optimizer=Adam(0.001), loss=Huber(delta=1.0))


    #  asta cred ca l am adaugat degeaba aici
    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               restore_best_weights=True)
    # partea de antrenare
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stop]
    )

    reconstructions = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)

    # Pragul de anomalie = percentila 95 din erorile pe clasa normală (0)
  
    error_threshold = np.percentile(reconstruction_error[y_test == 0], specified_threshold)

    y_pred = (reconstruction_error > error_threshold).astype(int)

    # EVALUARE

    print("Threshold for anomaly detection:", round(error_threshold, 5))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, reconstruction_error))

    # PLOT OPTIONAL  

    plt.figure(figsize=(10, 5))
    plt.hist(reconstruction_error[y_test == 0], bins=50, alpha=0.6, label="Normal")
    plt.hist(reconstruction_error[y_test == 1], bins=50, alpha=0.6, label="Attack")
    plt.axvline(error_threshold, color='red', linestyle='--', label="Threshold")
    plt.title("Distribuția erorii de reconstrucție")
    plt.xlabel("MSE")
    plt.ylabel("Număr de instanțe")
    plt.legend()
    plt.grid(True)
    plt.show()

possible_thresholds = [50, 55, 60, 65, 70, 75, 80, 85]
for threshold in possible_thresholds:
    print(f"For threshold {threshold}:")
    Autoencoder(threshold)