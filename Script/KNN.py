from load_files import training_set, testing_set
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances, classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from load_files import X_res, y_res, X_train, X_test, y_test


#  normalizam
# scaler = MinMaxScaler()
# training_set = pd.DataFrame(scaler.fit_transform(training_set), columns=training_set.columns)
# epochs = int(np.sqrt(len(training_set)))
epochs = 50
# NVM : nu am spatiu suficient pentru o matrice atat de mare
# dist_marix = pairwise_distances(training_set[:len(training_set)//2], metric = 'euclidean')
# print(dist_marix)

#  alegerea unui k potrivit reprezinta o dificultate in metoda KNN. De aceea, voi selecta valori random din range ul 0-50(am gasit cateva paperuri care
#  recomanda acest range: ai ss in obsidian).

def knn_anomaly_detection(X_train, X_test, y_test, n_epochs = 50, k_range = (1,50)):
    freq_of_anomalies = {}
    sample_size = min(1000, len(X_train))
    train_sample = X_train[np.random.choice(len(X_train), sample_size, replace=False)]

    # calculez distanta de la fiecare punct din dataset ul de test, la fiecare punct din setul de train
    all_distances = pairwise_distances(X_test, train_sample, metric="euclidean")


    for epoch in range(n_epochs):
        k = np.random.randint(*k_range)

        #  anomaly scores va fi o matrice ce va contine a k a distanta fata de fiecare punct
        #  din testing set
        anomaly_scores = np.partition(all_distances, k, axis=1)[:, k]

        threshold_for_anomalies = np.percentile(anomaly_scores, 95)

        # Get indices of anomalies
        anomaly_indices = np.where(anomaly_scores > threshold_for_anomalies)[0]
        #  [anomaly scores > threshold_for_anomalies] va returna a boolean mask pentru fiecare punct din anomaly_scores
        # apoi filtram ce puncte din testing data au anomaly score ul mai mare decat thresholdul


        # folosim un fel de vector de frecventa ca sa numaram de cate ori a fost
        # considerat fiecare punct ca anomalie
        for idx in anomaly_indices:
            freq_of_anomalies[idx] = freq_of_anomalies.get(idx, 0) + 1

    #  dupa ce am calculat distantele in functie de toti k, voi pastra punctele care apar cel mai des
    #  ca anomalii

    anomalies = sorted(freq_of_anomalies.items(), key=lambda x: x[1], reverse=True)
    top_70_percent = int(0.7 * len(anomalies))
    # final_anomalies = [i for i, count in anomalies[:top_70_percent]]
    top_anomalies = [idx for idx, count in anomalies if count > top_70_percent]

    # calculate accuracy
    # 5. Make predictions (1=anomaly, 0=normal)
    predictions = np.zeros(len(X_test))
    predictions[top_anomalies] = 1

    # 6. Evaluation
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return predictions

predictions = knn_anomaly_detection(X_res, X_test, y_test)