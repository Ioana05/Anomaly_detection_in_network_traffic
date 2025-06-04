import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from load_files import testing_set, X_res, y_res, X_test, y_test

def average_KNN_scikit(batch_size = 1000):
    step = 1000
    epochs = len(X_res)//10
    # in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)

    scores = np.zeros(len(X_test))
    steps = 0
    for k in range(step,  epochs, step):

        print(f"Epoch: {k}")
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_res)

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

average_KNN_scikit(batch_size=1000)