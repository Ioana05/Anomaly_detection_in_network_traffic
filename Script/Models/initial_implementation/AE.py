from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from Script.pipeline import load_and_preprocess_data
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

# filtram datele pentru ca vrem sa antrenam modelul doar pe date normale
X_train = X_train[y_train == 0]
y_train = y_train[y_train == 0]

def Autoencoder(specified_threshold):

    # primul strat din retea va avea numarul de neuroni egal cu numarul de feature uri din set astfel incat datele de intrare vor fi compatibile
    input_dim = X_train.shape[1] 

    # Arhitectura autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(4, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)    
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    # autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    autoencoder.compile(optimizer=Adam(0.001), loss=Huber(delta=1.0))

    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               restore_best_weights=True)
    # partea de antrenare
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stop]
    )

    reconstructions = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)
  
    error_threshold = np.percentile(reconstruction_error[y_test == 0], specified_threshold)

    y_pred = (reconstruction_error > error_threshold).astype(int)

    # Partea de evaluare
    print("Threshold for anomaly detection:", round(error_threshold, 5))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, reconstruction_error))


#  70 este cel mai bun threshold din cele incercate din intervalul [50, 75]
Autoencoder(70)