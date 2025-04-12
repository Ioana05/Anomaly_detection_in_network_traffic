from load_files import training_set, testing_set
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.metrics import accuracy_score

def preprocessing(dataset):
  dataset['proto'] = LabelEncoder().fit_transform(dataset['proto'])
  dataset['service'] = LabelEncoder().fit_transform(dataset['service'])
  dataset['state'] = LabelEncoder().fit_transform(dataset['state'])
  dataset['attack_cat'] = LabelEncoder().fit_transform(dataset['attack_cat'])

  return dataset

def normalization(dataset):
  scaler = MinMaxScaler()
  dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

  return dataset

training_set['proto'] = LabelEncoder().fit_transform(training_set['proto'])
training_set['service'] = LabelEncoder().fit_transform(training_set['service'])
training_set['state'] = LabelEncoder().fit_transform(training_set['state'])
training_set['attack_cat'] = LabelEncoder().fit_transform(training_set['attack_cat'])

# training_set['proto'] = LabelEncoder().fit_transform(training_set['proto'])
# training_set['service'] = LabelEncoder().fit_transform(training_set['service'])
# training_set['state'] = LabelEncoder().fit_transform(training_set['state'])
# training_set['attack_cat'] = LabelEncoder().fit_transform(training_set['attack_cat'])

# training_set_with_ids = training_set.copy()
# training_set = training_set.drop(columns = ['id'])

training_set = preprocessing(training_set)
testing_set = preprocessing(testing_set)

training_set_with_ids = training_set.copy()
testing_set_with_ids = testing_set.copy()

# training_set = training_set.drop(columns = ['id'])
# testing_set = testing_set.drop(columns = ['id'])

training_set = normalization(training_set)
testing_set = normalization(testing_set)

#  normalizam
# scaler = MinMaxScaler()
# training_set = pd.DataFrame(scaler.fit_transform(training_set), columns=training_set.columns)
epochs = int(np.sqrt(len(training_set)))
# NVM : nu am spatiu suficient pentru o matrice atat de mare
# dist_marix = pairwise_distances(training_set[:len(training_set)//2], metric = 'euclidean')
# print(dist_marix)

#  alegerea unui k potrivit reprezinta o dificultate in metoda KNN. De aceea, voi selecta valori random din range ul 0-50(am gasit cateva paperuri care
#  recomanda acest range: ai ss in obsidian).

# calculez distanta de la fiecare punct din dataset ul de test, la fiecare punct din setul de train
all_distances = pairwise_distances(testing_set, training_set, metric="euclidean")
freq_of_anomalies = {}

for epoch in range(epochs):
    k = np.random.randint(1, len(training_set))

    #  anomaly scores va fi o matrice ce va contine a k a distanta fata de fiecare punct
    #  din testing set
    anomaly_scores = np.partition(all_distances, k, axis=1)[:, k]

    threshold_for_anomalies = np.percentile(anomaly_scores, 30)

    #  [anomaly scores > threshold_for_anomalies] va returna a boolean mask pentru fiecare punct din anomaly_scores
    # apoi filtram ce puncte din testing data au anomaly score ul mai mare decat thresholdul
    possible_anomalies = testing_set_with_ids[anomaly_scores > threshold_for_anomalies]['id'].tolist()

    # folosim un fel de vector de frecventa ca sa numaram de cate ori a fost
    # considerat fiecare punct ca anomalie
    for i in possible_anomalies:
        if freq_of_anomalies.get(i) is None:
            freq_of_anomalies[i] = 0
        else:
            freq_of_anomalies[i] += 1

#  dupa ce am calculat distantele in functie de toti k, voi pastra punctele care apar cel mai des
#  ca anomalii

anomalies = sorted(freq_of_anomalies.items(), key=lambda x: x[1], reverse=True)
top_70_percent = int(0.7 * len(anomalies))
final_anomalies = [i for i, count in anomalies[:top_70_percent]]

# calculate accuracy
predictions = {}
for row in testing_set_with_ids.itertuples(index = True):
    if row.id in final_anomalies:
        predictions[row.id] = 1
    else:
        predictions[row.id] = 0


accuracy = accuracy_score(list(testing_set_with_ids['label']), list(predictions.values()))
print(accuracy)

accuracy = accuracy_score(list(testing_set_with_ids['label']), list(predictions.values()))
print(accuracy)
