import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

########## INCARC DATELE #####################
training_csv = ("C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_training-set.csv")
testing_csv = ("C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_testing-set.csv")

training_set = pd.read_csv(training_csv)
testing_set = pd.read_csv(testing_csv)

########## FEATURE ENGINEERING #################
def feature_engineering(dataset):
    dataset['network_bytes'] = dataset['sbytes'] + dataset['dbytes']
    dataset['bytes_ratio'] = dataset['sbytes'] / (dataset['dbytes'] + 1)
    dataset['total_bytes'] = dataset['sbytes'] + dataset['dbytes']
    dataset['duration_per_byte'] = dataset['dur'] / (dataset['total_bytes'] + 1)
    dataset['pkts_per_sec'] = dataset['spkts'] / (dataset['dur'] + 1)
    dataset['direction_flag'] = (dataset['spkts'] > dataset['dpkts']).astype(int)

    return dataset

########## ANALIZAM CORELATII ######################
import seaborn as sns
import matplotlib.pyplot as plt

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

#################### RENUNTAM LA COLOANELE CARE AU CORELATII MARI ###################
training_set = training_set.drop(columns = columns_to_drop)
testing_set = testing_set.drop(columns = columns_to_drop)

# Înlocuiește "-" cu "None"
for column in training_set.columns:
  training_set[column] = training_set[column].apply(lambda x:"None" if x=="-" else x)
  testing_set[column] = testing_set[column].apply(lambda x:"None" if x=="-" else x)

################### ENCODARE PE VALORILE NON NUMERICE #############################
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


################### SKRRR ######################
def apply_log1p_if_skewed(df, threshold=1.0):
    skewed_feats = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_cols = skewed_feats[skewed_feats > threshold].index

    df[skewed_cols] = np.log1p(df[skewed_cols])
    return df, list(skewed_cols)


################## NORMALIZAM ###################
def normalization(dataset, fit=True, scaler=None):
    if fit:
        scaler = MinMaxScaler()
        dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
        return dataset, scaler
    else:
        dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)
        return dataset


################# ECHILIBRAM DATASETUL ###############
# echilibram datasetul
def balance(dataset):
  anomalies = dataset[dataset['label'] == 1]
  normal_points = dataset[dataset['label']== 0]
  print("Anomalies:", len(anomalies))
  print("Normal points:", len(normal_points))
  anomalies = anomalies.sample(n = min(len(anomalies), len(normal_points)))
  normal_points = normal_points.sample(n = min(len(anomalies), len(normal_points)))

  balanced_dataset = pd.concat([anomalies, normal_points], ignore_index=True)
  shuffled_dataset = balanced_dataset.sample(frac = 1).reset_index(drop = True)
  shuffled_dataset['id'] = shuffled_dataset.index
  print(len(shuffled_dataset))

  # shuffled_dataset = shuffled_dataset[:6000]

  return shuffled_dataset

############### APLIC TOATE FUNCTIILE ################
training_set = balance(training_set)
testing_set = balance(testing_set)

training_set, skewed_cols = apply_log1p_if_skewed(training_set)
testing_set, skewed_cols = apply_log1p_if_skewed(testing_set)

training_set = preprocessing(training_set)
testing_set = preprocessing(testing_set)

final_training_set, scaler = normalization(training_set, fit=True)
final_testing_set = normalization(testing_set, fit=False, scaler=scaler)

epochs = int(np.sqrt(len(training_set)))

# Setezi opțiunea globală pentru a afișa toate coloanele
pd.set_option('display.max_columns', None)

# Afișezi primele 5 rânduri (sau câte vrei tu)
# print(training_set.head())
# print(testing_set.head())

epochs = int(np.sqrt(len(training_set)))
sub_sampling_size = 256
number_of_trees = 200


scaler = StandardScaler()
df_scaled = scaler.fit_transform(testing_set.drop(columns=['label', 'id', 'proto', 'service', 'state']))

#try to use PCA to reduce dimensions and plot the dataset
# pca = PCA(n_components=2)
# df_pca = pca.fit_transform(df_scaled)
#
# plt.scatter(df_pca[:,0], df_pca[:, 1], cmap = "coolwarm", alpha = 0.5)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Projection of Multi-Dimensional Data")
# plt.show()