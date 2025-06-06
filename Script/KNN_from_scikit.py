import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from load_files import change_proportion_of_data, load_and_preprocess_data
X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )


training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.05
test_anomalies = 0.1
# change proportion
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)
# testing_set = change_proportion_of_data(testing_set_resampled, percentage_anomalies=test_anomalies, total=30000)

# After changing proportions:
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 

X_test = testing_set_resampled.drop(columns=['label'])              
y_test = testing_set_resampled['label']           


def average_KNN_scikit(batch_size = 1000):
    step = 1000
    epochs = len(X_train)//10
    # in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)

    scores = np.zeros(len(X_test))
    steps = 0
    for k in range(step,  epochs, step):

        print(f"Epoch: {k}")
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_train)

        for start in range(0, len(X_test), batch_size):
            end = min(start + batch_size, len(X_test))
            current_batch = X_test[start:end]

            distances, _ = knn.kneighbors(current_batch)
            average_distances = np.mean(distances, axis = 1)
            scores[start:end] += average_distances
        
        steps += 1 

    
    average_scores = scores/steps

    # cele mai mari 5% distante vor fi considerate anomalii
    threshold_for_anomalies = np.percentile(average_scores, 95)

    predictions = (average_scores > threshold_for_anomalies )

    print(classification_report(y_test, predictions))


def average_KNN_scikitV2(batch_size = 1000):
    # in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)
    k_values = np.linspace(int(0.05 * len(X_train)), int(0.1 * len(X_train)), num=10, dtype=int)
    scores = np.zeros(len(X_test))

    for k in k_values:

        print(f"Epoch: {k}")
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_train)

        for start in range(0, len(X_test), batch_size):
            end = min(start + batch_size, len(X_test))
            current_batch = X_test[start:end]

            distances, _ = knn.kneighbors(current_batch)
            average_distances = np.mean(distances, axis = 1)
            scores[start:end] += average_distances

    
    average_scores = scores/len(k_values)

    # cele mai mari 5% distante vor fi considerate anomalii
    threshold_for_anomalies = np.percentile(average_scores, 95)

    predictions = (average_scores > threshold_for_anomalies )

    print(classification_report(y_test, predictions))



def average_KNN_scikit_rule_thumb(batch_size = 1000):
   
    # in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)

    scores = np.zeros(len(X_test))
    # we use rule of thumb
    k = int(np.sqrt(len(X_train)))

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)

    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        current_batch = X_test[start:end]

        distances, _ = knn.kneighbors(current_batch)
        kth_distances = distances[:, -1]
        scores[start:end] += kth_distances


    # cele mai mari 5% distante vor fi considerate anomalii
    threshold_for_anomalies = np.percentile(scores, 50)

    predictions = (scores > threshold_for_anomalies )

    print(classification_report(y_test, predictions))


from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import numpy as np

def k_random_voting_methods(batch_size = 1000):
    freq_of_anomalies = {}
    epochs = 7
    print(f"Total epochs: {epochs}")
    # construim o lista cu 7 valori pt k distribuite uniform
    k_values = np.linspace(int(0.05 * len(X_train)), int(0.1 * len(X_train)), num=7, dtype=int)
    for epoch in range(epochs):
        print(f"Current epoch: {epoch}")
        knn = NearestNeighbors(n_neighbors=k_values[epoch])
        knn.fit(X_train)
        distances, _ = knn.kneighbors(X_test)

        anomaly_scores = distances[:, -1]
        threshold_for_anomalies = np.percentile(anomaly_scores, 50)

        possible_anomalies = [i for i, distance in enumerate(anomaly_scores) if distance > threshold_for_anomalies]

        for i in possible_anomalies:
            freq_of_anomalies[i] = freq_of_anomalies.get(i, 0) + 1

    counts = list(freq_of_anomalies.values())
    if len(counts) == 0:
        print("No anomalies detected.")
        return

    threshold = np.percentile(counts, 50)

    scores = np.zeros(len(X_test))
    for idx, count in freq_of_anomalies.items():
        if count > threshold:
            scores[idx] = 1

    print(classification_report(y_test, scores))


def NN(batch_size = 1000):
    # in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)

    scores = np.zeros(len(X_test))
    # we use rule of thumb
    k = 2

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)

    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        current_batch = X_test[start:end]

        distances, _ = knn.kneighbors(current_batch)
        kth_distances = distances[:, -1]
        scores[start:end] += kth_distances


    # cele mai mari 5% distante vor fi considerate anomalii
    threshold_for_anomalies = np.percentile(scores, 50)

    predictions = (scores > threshold_for_anomalies )

    print(classification_report(y_test, predictions))


        
# k_random_voting_methods()
# NN()
# average_KNN_scikit_rule_thumb(batch_size=1000)
average_KNN_scikitV2()