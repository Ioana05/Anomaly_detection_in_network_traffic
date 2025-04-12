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

def KNN(training_set, current_point, k):

    # compute distance from
    distances = pairwise_distances(training_set,[current_point], metric= "euclidean")

    # the scored point is not included among the k-nearest neighbors to exclude overfitting
    distances = distances[distances > 0]

    #so we want the kth smallest distance
    # pentru asta voi folosi metoda din numpy, metoda din scikit
    # asemanatoare ar fi fost NearestNeighbors dar foloseste concepte mai complexe
    # pentru eficientizare

    # varianta din numpy pe care am folosit o foloseste 'introselect' ca metoda de sortare
    # complexitatea Worst Case este O(n), work space este 0 , iar algoritmul nu este stable


    anomaly_score = np.partition(distances, k-1)[k-1]

    return anomaly_score

#  alegerea unui k potrivit reprezinta o dificultate in metoda KNN. De aceea, voi selecta valori random din range ul 0-50(am gasit cateva paperuri care
#  recomanda acest range: ai ss in obsidian).


# in mai multe paperuri(sunt alesi k intre 5 si 10% din numarul total de date)


freq_of_anomalies = {}
for epoch in range(epochs):
    k = np.random.randint(0, len(training_set))
    # distances va contine distanta fiecarui punct fata de al k lea vecin al sau
    distances = []
    for i,current_point in testing_set.iterrows():
        distance = KNN(training_set, current_point, k)
        distances.append(distance)

    # cele mai mari 5% distante vor fi considerate anomalii
    threshold_for_anomalies = np.percentile(distances, 30)

    possible_anomalies = [i for i, distance in enumerate(distances) if distance > threshold_for_anomalies]

    # folosim un fel de vector de frecventa ca sa numaram de cate ori a fost
    # considerat fiecare punct ca anomalie
    for i in possible_anomalies:
        if freq_of_anomalies.get(i) is None:
            freq_of_anomalies[i] = 0
        else:
            freq_of_anomalies[i] += 1


#  dupa ce am calculat distantele in functie de toti k, voi pastra punctele care apar cel mai des
#  ca anomalii

anomalies = sorted(freq_of_anomalies.items(), key = lambda x:x[1], reverse=True)
top_70_percent = int(0.7 * len(anomalies))
final_anomalies = [i for i, count in anomalies[top_70_percent:]]


# calculate accuracy
predictions = {}
for row in testing_set_with_ids.itertuples(index = True):
    if row.id in final_anomalies:
        predictions[row.id] = 1
    else:
        predictions[row.id] = 0


accuracy = accuracy_score(list(testing_set_with_ids['label']), list(predictions.values()))
print(accuracy)
