from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from load_files import load_and_preprocess_data, change_proportion_of_data
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )
training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
test_anomalies = 0.1
# change proportion
# training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)
# testing_set_resampled = change_proportion_of_data(testing_set_resampled, percentage_anomalies=test_anomalies, total=30000)

# After changing proportions:
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 

X_test = testing_set_resampled.drop(columns=['label'])              
y_test = testing_set_resampled['label']        

X_train = X_train[y_train == 0]
y_train = y_train[y_train == 0]

def Autoencoder():

    input_dim = X_train.shape[1]  # după SMOTE/feature selection

    # Arhitectura autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = LeakyReLU(alpha=0.01)(encoded) 
    encoded = Dense(64, activation='relu')(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded) 
    encoded = Dense(32, activation='relu')(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded) 
    bottleneck = Dense(8, activation='relu')(encoded)
    bottleneck = LeakyReLU(alpha=0.01)(bottleneck) 
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = LeakyReLU(alpha=0.01)(decoded) 
    decoded = Dense(64, activation='relu')(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    
    decoded = Dense(128, activation='relu')(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded) 
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    #  asta cred ca l am adaugat degeaba aici
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # partea de antrenare
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )

    reconstructions = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)

    # Pragul de anomalie = percentila 95 din erorile pe clasa normală (0)
    error_threshold = np.percentile(reconstruction_error[y_test == 0], 70)

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


Autoencoder()