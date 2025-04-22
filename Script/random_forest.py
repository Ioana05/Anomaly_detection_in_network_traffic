import random

from load_files import training_set, testing_set
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# actually mi am dat seama ca nu cred ca nu este o idee buna sa folosesc one hot encoder pentru ca vor exista coloane cu o singura valoare de 1 swi restul 0,
# iar asta s ar putea sa influenteze resultatul pentru isolation forest


def feature_engineering(dataset):
    dataset['network_bytes'] = dataset['sbytes'] + dataset['dbytes']
    dataset['bytes_ratio'] = dataset['sbytes'] / (dataset['dbytes'] + 1)
    dataset['total_bytes'] = dataset['sbytes'] + dataset['dbytes']
    dataset['duration_per_byte'] = dataset['dur'] / (dataset['total_bytes'] + 1)
    dataset['pkts_per_sec'] = dataset['spkts'] / (dataset['dur'] + 1)
    dataset['direction_flag'] = (dataset['spkts'] > dataset['dpkts']).astype(int)

    return dataset
encoders = {}
#  vom folosi acest dictionar de encoders ca sa fim siguri ca se vor face aceleasi encodari si pe datele de test si pe cele de train
def preprocessing(dataset, fit = True):
  categorical_columns = ['proto', 'service', 'state']
  for col in categorical_columns:
      if fit:
          encoders[col] = LabelEncoder()
          dataset[col] = encoders[col].fit_transform(dataset[col])
      else:
          dataset[col] = encoders[col].transform(dataset[col])
  return dataset
#  corelatii

# o sa consider anomalie mare tot ce e peste 0.9

# drop classes which are not useful for the classification
training_set.drop(['attack_cat'], axis=1, inplace=True)
testing_set.drop(['attack_cat'], axis=1, inplace=True)

# selectează doar coloanele numerice
numeric_training_set = training_set.select_dtypes(include=['number'])
numeric_testing_set = testing_set.select_dtypes(include=['number'])

# calculează matricea de corelație
corr_matrix_test = numeric_training_set.corr().abs()
corr_matrix_train = numeric_testing_set.corr().abs()

# heatmap pentru vizualizare
# plt.figure(figsize=(25, 10))
# sns.heatmap(corr_matrix_test, annot=True, cmap='coolwarm')
# plt.title("Matricea de corelație pe datele de test")
# plt.show()

# heatmap pentru vizualizare
# plt.figure(figsize=(25, 10))
# sns.heatmap(corr_matrix_train, annot=True, cmap='coolwarm')
# plt.title("Matricea de corelație pe datele de train")
# plt.show()

# pastram doar corelatiile de deasupra diagonalei principale
upper_values_train = corr_matrix_train.where(np.triu(np.ones(corr_matrix_train.shape), k=1).astype(bool))
# upper_values_train = corr_matrix_train.where(np.triu(np.ones(corr_matrix_train.shape), k=1).astype(bool))
high_corr_pairs = [
    (col, row, upper_values_train.loc[col, row])
    for col in upper_values_train.columns
    for row in upper_values_train.index
    if upper_values_train.loc[col, row] > 0.95
]
columns_to_drop = set()
for col, row, values in high_corr_pairs:
  if col not in columns_to_drop:
    columns_to_drop.add(col)

training_set = feature_engineering(training_set)
testing_set = feature_engineering(testing_set)

columns_to_drop = list(columns_to_drop)

#  eliminam coloanele care au corelatie mare
training_set = training_set.drop(columns = columns_to_drop)
testing_set = testing_set.drop(columns = columns_to_drop)

# Înlocuiește "-" cu "None"
for column in training_set.columns:
  training_set[column] = training_set[column].apply(lambda x:"None" if x=="-" else x)
  testing_set[column] = testing_set[column].apply(lambda x:"None" if x=="-" else x)


def apply_log1p_if_skewed(df, threshold=1.0):
    skewed_feats = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_cols = skewed_feats[skewed_feats > threshold].index

    df[skewed_cols] = np.log1p(df[skewed_cols])
    return df, list(skewed_cols)


def normalization(dataset):
  scaler = MinMaxScaler()
  # dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

  cols_to_normalize = dataset.drop(columns=['label', 'id']).columns
  dataset[cols_to_normalize] = scaler.fit_transform(dataset[cols_to_normalize])

  return dataset

# echilibram datasetul
def balance(dataset):
  anomalies = dataset[dataset['label'] == 1]
  normal_points = dataset[dataset['label'] == 0]
  anomalies = anomalies.sample(n = min(len(anomalies), len(normal_points)))
  normal_points = normal_points.sample(n = min(len(anomalies), len(normal_points)))

  balanced_dataset = pd.concat([anomalies, normal_points], ignore_index=True)
  shuffled_dataset = balanced_dataset.sample(frac = 1).reset_index(drop = True)
  shuffled_dataset['id'] = shuffled_dataset.index

  return shuffled_dataset

#  balansam seturile de date pentru ca in momentul de fata ave 70%din set anomalii
training_set = balance(training_set)
testing_set = balance(testing_set)

training_set, skewed_cols = apply_log1p_if_skewed(training_set)
testing_set, skewed_cols = apply_log1p_if_skewed(testing_set)


#  preprocesam setul de date
training_set = preprocessing(training_set)
testing_set = preprocessing(testing_set)

# training_set_with_ids = training_set.copy()
# testing_set_with_ids = testing_set.copy()

# training_set = training_set.drop(columns = ['id'])
# testing_set = testing_set.drop(columns = ['id'])

#  normalizam
training_set = normalization(training_set)
testing_set = normalization(testing_set)

# cam atat cu datasetul, vezi mai tarziu daca trebuie sa normalizezi datele(s ar putea sa fie nevoie)
# so we will try to con vert the categorical values in numerical values using
# LabelEncoder which will convert the non-numerical values in integer values

# average path length is given by the estimation of average height for BST
def calculateC(n):
    gamma = 0.5772156649
    if n <= 1:
        return 0
    expected_average_path_length = 2.0*(math.log(n-1) + gamma) - (2.0*(n-1)/(n*1.0))
    return expected_average_path_length

def iTree(subset, current_tree_height, limit_of_height):
    # i m not sure if I need the second condition in if
    if current_tree_height >= limit_of_height or len(subset) <= 1:
        # see why should I return a dictionary
        #  nu cred ca e ok ceea ce atribui frunnzelor in randul de mai jos
        return {"Size": len(subset)}
    else:
        attribute_names = pd.DataFrame(subset).columns
        # we should exclude some columns from the dataset when choosing the random attribute, like id, state
        random_attr = attribute_names[random.randint(0, len(attribute_names)-1)]
        while random_attr == 'id' or random_attr == 'state' or random_attr == 'label':
          random_attr = attribute_names[random.randint(0, len(attribute_names)-1)]
        # find min and max to know the interval where to find a splitting value
        min_value = subset[random_attr].min()
        max_value = subset[random_attr].max()

        # now we are randomly selecting a split point
        random_split_value = random.uniform(min_value, max_value)

        lower_values = subset[subset[random_attr] < random_split_value]
        greater_values = subset[subset[random_attr] >= random_split_value]

        if len(lower_values) == 0 or len(greater_values) == 0:
            return {"Size": len(subset)}

        return {"Left": iTree(lower_values, current_tree_height + 1, limit_of_height),
                "Right": iTree(greater_values, current_tree_height + 1, limit_of_height),
                "SplitAtt": random_attr,
                "SplitValue": random_split_value
                }

def iForest(training_set, number_of_trees, sub_sampling_size):
    height_limit = math.ceil(math.log(sub_sampling_size))
    forest = []
    for i in range(number_of_trees):
        # sub_sampling_set = training_set.sample(sub_sampling_size).reset_index(drop = True)
        # sub_sampling_set = sub_sampling_set.drop(columns=['label', 'id', 'attack_cat', 'service', 'proto', 'state', 'is_ftp_login', 'ct_ftp_cmd'])  # adaugă linia asta
        anomalies = training_set[training_set['label'] == 1].sample(n = sub_sampling_size//2)
        normal_points = training_set[training_set['label'] == 0].sample(n=sub_sampling_size // 2)
        # reset index is to shuffle the dataset
        sub_sampling_set = pd.concat([anomalies, normal_points]).reset_index(drop = True)
        isolation_Tree = iTree(sub_sampling_set, 0, height_limit)
        forest.append(isolation_Tree)
    return forest


def pathLength(dataPoint, Tree, current_path_length):
    #  verificam daca am ajuns intr-o frunza a arborelui
    if 'Size' in Tree:
        return current_path_length + calculateC(Tree['Size'])
    split_attribute = Tree['SplitAtt']
    if dataPoint[split_attribute] < Tree['SplitValue']:
        return pathLength(dataPoint, Tree['Left'], current_path_length + 1)
    else:
        return pathLength(dataPoint, Tree['Right'], current_path_length + 1)

def computeAnomalyScore(dataPoint, forest):
    sum_of_paths = 0
    for tree in forest:
        path_length = pathLength(dataPoint, tree, 0)
        sum_of_paths += path_length
    average_path_length = sum_of_paths/len(forest)
    anomaly_score = 2**(-(average_path_length)/calculateC(sub_sampling_size[0]))
    return anomaly_score

number_of_trees = [100]
sub_sampling_size = [250]

for epoch in range(len(number_of_trees)):

    forest = iForest(training_set, number_of_trees[epoch], sub_sampling_size[epoch])
    classified_labels = {}
    scores = []

    for i, x in testing_set.iterrows():
        result = computeAnomalyScore(x, forest)
        scores.append(result)
    threshold = np.percentile(scores, 40)

    for i, x in testing_set.iterrows():
        if scores[i] > 0.47:
            classified_labels[i] = float(1.0)
        else:
            classified_labels[i] = float(0.0)

    print(f"Number of trees: {number_of_trees[epoch]} and sub-sampling size: {sub_sampling_size}")
    print(accuracy_score(testing_set['label'], list(classified_labels.values())))


import matplotlib.pyplot as plt

scores = [computeAnomalyScore(x, forest) for _, x in testing_set.iterrows()]
anomalies = [score for score, is_anomaly in zip(scores, testing_set['label']) if is_anomaly]
normal_points = [score for score, is_anomaly in zip(scores, testing_set['label']) if not is_anomaly]
print(len(anomalies))
print(len(normal_points))

# Creează histograma
plt.hist(normal_points, bins=100, alpha=0.7, label='Non-anomalies', color='blue')
plt.hist(anomalies, bins=100, alpha=0.7, label='Anomalies', color='red')

# Adaugă titlul și legenda
plt.title("Distribuție scoruri de anomalie")
plt.legend()

# Afișează graficul
plt.show()

# plt.hist(scores, bins=100)
# plt.title("Distribuție scoruri de anomalie")
# plt.show()

