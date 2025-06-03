from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from load_files import X_res, y_res, X_test, y_test
import numpy as np



def Autoencoder():

    input_dim = X_res.shape[1]  # după SMOTE/feature selection

    # Arhitectura autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    bottleneck = Dense(16, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(bottleneck)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    X_normal = X_res[y_res == 0]
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=60,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )


    reconstructions = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)

    # Pragul de anomalie = percentila 95 din erorile pe clasa normală (0)
    error_threshold = np.percentile(reconstruction_error[y_test == 0], 80)

    # Predictii: 1 = anomalie, 0 = normal
    y_pred = (reconstruction_error > error_threshold).astype(int)

    # ===================== EVALUARE ======================== #

    print(" Threshold for anomaly detection:", round(error_threshold, 5))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, reconstruction_error))

    # ===================== PLOT OPTIONAL ======================== #

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
