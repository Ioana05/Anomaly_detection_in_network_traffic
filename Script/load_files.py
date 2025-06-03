import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, f1_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNetCV


########## INCARC DATELE #####################
training_csv = ("C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_training-set.csv")
testing_csv = ("C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_testing-set.csv")

training_set = pd.read_csv(training_csv)
testing_set = pd.read_csv(testing_csv)

########## FEATURE ENGINEERING #################
def feature_engineering(dataset):
    # Basic features
    # dataset['network_bytes'] = dataset['sbytes'] + dataset['dbytes']
    # dataset['bytes_ratio'] = dataset['sbytes'] / (dataset['dbytes'] + 1)
    # dataset['total_pkts'] = dataset['spkts'] + dataset['dpkts']
    # dataset['pkts_ratio'] = dataset['spkts'] / (dataset['dpkts'] + 1)
    # dataset['duration_per_byte'] = dataset['dur'] / (dataset['network_bytes'] + 1)
    # dataset['pkts_per_sec'] = dataset['total_pkts'] / (dataset['dur'] + 1)
    # dataset['direction_flag'] = (dataset['spkts'] > dataset['dpkts']).astype(int)

    # # Advanced features
    # dataset['avg_pkt_size'] = dataset['network_bytes'] / (dataset['total_pkts'] + 1)
    # dataset['flow_byte_rate'] = dataset['network_bytes'] / (dataset['dur'] + 1)
    # dataset['flow_pkt_rate'] = dataset['total_pkts'] / (dataset['dur'] + 1)
    # dataset['tcp_flag_count'] = dataset[['ct_flw_http_mthd', 'is_ftp_login']].sum(axis=1)


    # return dataset

    dataset['flow_iat_std'] = dataset[['smean', 'dmean']].std(axis=1)  # Inter-arrival time variability
    dataset['proto_service_combo'] = dataset['proto'].astype(str) + "_" + dataset['service'].astype(str)  # Interaction feature
    dataset['packet_size_entropy'] = -(dataset['spkts']*np.log(dataset['spkts']+1e-6) + dataset['dpkts']*np.log(dataset['dpkts']+1e-6))  # Traffic randomness
    return dataset

################### ENCODARE PE VALORILE NON NUMERICE #############################
encoders = {}
#  vom folosi acest dictionar de encoders ca sa fim siguri ca se vor face aceleasi encodari si pe datele de test si pe cele de train
def preprocessing(dataset, fit = True):
  categorical_columns = ['proto', 'service', 'state', 'proto_service_combo']
  if fit:
      # Fit encoders on training data only
      for col in categorical_columns:
          encoders[col] = LabelEncoder().fit(dataset[col])
  # Always transform using fitted encoders
  for col in categorical_columns:
      dataset[col] = encoders[col].transform(dataset[col])
  return dataset

################### CHECK THE SKEWING OF THE DATASET  ######################
def apply_log1p_if_skewed(df, threshold=1.0):
    skewed_feats = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_cols = skewed_feats[skewed_feats > threshold].index

    df[skewed_cols] = np.log1p(df[skewed_cols])
    return df, list(skewed_cols)


################## NORMALIZAM ###################
def normalization(dataset, fit=True, scaler=None):
    dataset = dataset.select_dtypes(include=["number"])
    if fit:
        scaler = RobustScaler()
        features = scaler.fit_transform(dataset)
        return features, scaler
    else:
        features = scaler.transform(dataset)
        return features, scaler


################# ECHILIBRAM DATASETUL ###############
# echilibram datasetul
def balance(dataset, min_samples=100, max_samples=3000, random_state=42):
    if 'attack_cat' not in dataset.columns:
        raise ValueError("Column 'attack_cat' is required for stratified balancing.")

    normal = dataset[dataset['label'] == 0]
    anomalies = dataset[dataset['label'] == 1]
    attack_types = anomalies['attack_cat'].unique()

    sampled_anomalies = []
    for attack in attack_types:
        subset = anomalies[anomalies['attack_cat'] == attack]
        count = min(max(len(subset), min_samples), max_samples)
        sampled = subset.sample(n=min(count, len(subset)), random_state=random_state)
        sampled_anomalies.append(sampled)

    balanced_anomalies = pd.concat(sampled_anomalies, ignore_index=True)
    balanced_normals = normal.sample(n=len(balanced_anomalies), random_state=random_state, replace = False)

    final = pd.concat([balanced_anomalies, balanced_normals], ignore_index=True).sample(frac=1, random_state=random_state)
    final['id'] = final.index
    print(f"Balanced (semi-stratified) dataset size: {len(final)}")

    return final

def stratified_kfold_balance(dataset, n_splits=5, min_samples=100, max_samples=3000, random_state=42):

    if 'attack_cat' not in dataset.columns:
        raise ValueError("Column 'attack_cat' is required for stratified balancing.")

    # First create balanced dataset using your existing logic
    balanced_data = balance(dataset, min_samples, max_samples, random_state)

    # Create a combined stratification column (attack_cat + label)
    balanced_data['stratify_col'] = balanced_data['attack_cat'].astype(str) + "_" + balanced_data['label'].astype(str)

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Generate folds
    folds = []
    for train_idx, test_idx in skf.split(balanced_data, balanced_data['stratify_col']):
        folds.append((train_idx, test_idx))

    return folds, balanced_data


def elastic_net(X_train, y_train, X_test, y_test):
  # Separate features and label
  X_train = training_set.drop(columns=['attack_cat'])

  # Use ElasticNetCV for feature selection
  elastic_net = ElasticNetCV(l1_ratio=[0.7, 0.9, 1],  # More L1 emphasis
                       cv=5,
                       selection='random',  # Better for feature selection
                       tol=1e-3)
  elastic_net.fit(X_train, y_train)

  # Get features with non-zero coefficients
  selected_features = X_train.columns[elastic_net.coef_ != 0].tolist()
  print(f"Selected features: {selected_features}")

  # Filter both training and testing sets to use only selected features
  training_set_filtered = X_train[selected_features]
  testing_set_filtered = X_test[selected_features]
  return training_set_filtered, testing_set_filtered

########## ANALIZAM CORELATII ######################

def corelation_matrix():
  # drop classes which are not useful for the classification
  # training_set.drop(['attack_cat'], axis=1, inplace=True)
  # testing_set.drop(['attack_cat'], axis=1, inplace=True)

  # selectează doar coloanele numerice
  numeric_training_set = training_set.select_dtypes(include=['number'])
  numeric_testing_set = testing_set.select_dtypes(include=['number'])

  # calculează matricea de corelație
  corr_matrix_test = numeric_testing_set.corr().abs()
  corr_matrix_train = numeric_training_set.corr().abs()

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

  return list(columns_to_drop)



# Dupa o analiza manuala a setului de date, am observat ca exista multe -
# in coloana SERVICE
# Calculăm moda pe setul de training (nu pe setul de test)
def clean_dataset():
    for col in training_set.columns:
        # doar pe coloana Sevice intalnim aceste '-' simboluri pe care vrem sa le inlocuim
        if col == 'service':
            # inlocuiesc '-' cu moda
            mode_val = training_set[col].mode()[0]
            training_set[col] = training_set[col].replace('-', mode_val)
            testing_set[col] = testing_set[col].replace('-', mode_val)

        # verific daca exista si alte coloane cu None
        if training_set[col].isnull().any():
            if training_set[col].dtype == 'object':
                mode_val = training_set[col].mode()[0]
                training_set[col] = training_set[col].fillna(mode_val)
                testing_set[col] = testing_set[col].fillna(mode_val)
            else:
                median_val = training_set[col].median()
                training_set[col] = training_set[col].fillna(median_val)
                testing_set[col] = testing_set[col].fillna(median_val)
    return training_set

def remove_anomalies(X, y, percentage=0.3):

    if not 0 <= percentage < 1:
        raise ValueError("Percentage must be between 0 and 1")

    # Separate normal and anomalous samples
    normal_X = X[y == 0]
    anomalous_X = X[y == 1]
    normal_y = y[y == 0]
    anomalous_y = y[y == 1]

    # Calculate the number of anomalies to remove
    num_anomalies_to_remove = int(len(anomalous_X) * percentage)

    if num_anomalies_to_remove > 0:
        # Randomly sample anomalies to remove
        drop_indices = np.random.choice(anomalous_X.index, num_anomalies_to_remove, replace=False)
        anomalous_X_filtered = anomalous_X.drop(drop_indices)
        anomalous_y_filtered = anomalous_y.drop(drop_indices)

        # Combine the filtered anomalies with the normal samples
        X_filtered = pd.concat([normal_X, anomalous_X_filtered])
        y_filtered = pd.concat([normal_y, anomalous_y_filtered])
    else:
        X_filtered = X
        y_filtered = y

    return X_filtered, y_filtered

# curatarea setului de date
training_set = clean_dataset()

# Feature engineering
training_set = feature_engineering(training_set)
testing_set = feature_engineering(testing_set)

# encode categorical features
training_set = preprocessing(training_set)
testing_set = preprocessing(testing_set)

# obtinem coloanele care ar putea fi eliminate calculand matricea de corelatie
columns_to_drop = corelation_matrix()

# RENUNTAM LA COLOANELE CARE AU CORELATII MARI
training_set = training_set.drop(columns = columns_to_drop)
testing_set = testing_set.drop(columns = columns_to_drop)

# drop classes which are not useful for the classification
# training_set.drop(['attack_cat'], axis=1, inplace=True)
# testing_set.drop(['attack_cat'], axis=1, inplace=True)


# separate features and labels
X_train = training_set.drop(columns=['label', 'id'])
y_train = training_set['label']
X_test = testing_set.drop(columns=['label', 'id'])
y_test = testing_set['label']

# # 5. Apply log transform to skewed features
# X_train, skewed_cols = apply_log1p_if_skewed(X_train.copy())  # Apply to a copy to avoid modifying the original
# X_test_copy = X_test.copy() # Create a copy of X_test to apply log transform
# X_test_copy[skewed_cols] = np.log1p(X_test_copy[skewed_cols])
# X_test = X_test_copy # Assign the modified X_test back

# X_train, y_train = remove_anomalies(X_train, y_train, percentage=0.8)


#  Feature scaling
X_train, scaler = normalization(X_train, fit=True)
X_test, scaler_test = normalization(X_test, fit=False, scaler=scaler)


#  Feature selection
selector = SelectKBest(f_classif, k=30)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
# X_train, X_test = elastic_net(X_train,y_train, X_test, y_test)

#  Handle class imbalance
RANDOM_STATE = 42
smote = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X_train, y_train)
# X_res =X_train
# y_res = y_train

# training_set = balance(training_set)
# testing_set = balance(testing_set)


# Setezi opțiunea globală pentru a afișa toate coloanele
pd.set_option('display.max_columns', None)


# Afișezi primele 5 rânduri (sau câte vrei tu)
# print(training_set.head())
# print(testing_set.head())