import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNetCV
TRAINING_CSV_PATH = "C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_training-set.csv"
TESTING_CSV_PATH = "C:/Users/Asus/Desktop/Anomaly_detection_in_network_traffic/datasets/UNSW_NB15_testing-set.csv"


# am incercat si sa antrenez modelele pe un singur tip de atac (rezultatele au fost mai slabe)
def filter_dataset_by_attack(df, attack_type_to_keep):
    
    # filtram traficul normal
    normal_traffic = df[df['label'] == 0]
    # filtram tipul de atac specificat
    specific_attack_traffic = df[(df['label'] == 1) & (df['attack_cat'] == attack_type_to_keep)]
    # le concatenam
    filtered_df = pd.concat([normal_traffic, specific_attack_traffic]).reset_index(drop=True)
    print(f"Filtered dataset to include only '{attack_type_to_keep}' and normal traffic. New size: {len(filtered_df)}")
    return filtered_df


# Dupa o analiza manuala a setului de date, am observat ca exista multe -
# in coloana SERVICE
# Calculăm moda pe setul de training (nu pe setul de test)
def clean_dataset(dataset):
    for col in dataset.columns:
        # doar pe coloana Sevice intalnim aceste '-' simboluri pe care vrem sa le inlocuim
        if col == 'service':
            # inlocuiesc '-' cu moda
            mode_val = dataset[col].mode()[0]
            dataset[col] = dataset[col].replace('-', mode_val)

        # verific daca exista si alte coloane cu None
        if dataset[col].isnull().any():
            if dataset[col].dtype == 'object':
                mode_val = dataset[col].mode()[0]
                dataset[col] = dataset[col].fillna(mode_val)
            else:
                median_val = dataset[col].median()
                dataset[col] = dataset[col].fillna(median_val)
    return dataset

########## FEATURE ENGINEERING #################
def feature_engineering(dataset):

     # adaugam o coloana care sa contina deviatia standard a timpilor de sosire
    dataset['flow_iat_std'] = dataset[['smean', 'dmean']].std(axis=1)
    # combinatie dintre protocolul utilizat si serviciu (ne ofera mai multe detalii despre ce actiune a avut loc(de ex TCP+HTTP ar putea indica web searching))
    dataset['proto_service_combo'] = dataset['proto'].astype(str) + "_" + dataset['service'].astype(str)
    # calculeaza entropia dimensiunii pachetelor (practic cat de random e traficul)
    dataset['packet_size_entropy'] = -(dataset['spkts']*np.log(dataset['spkts']+1e-6) + dataset['dpkts']*np.log(dataset['dpkts']+1e-6))
    # dataset.drop(columns = ['proto', 'service', 'spkts', 'dpkts'])

    return dataset

################### ENCODARE PE VALORILE NON NUMERICE ##############
encoders = {}

def preprocessing(dataset, fit=True):
    categorical_columns = ['proto', 'service', 'state', 'proto_service_combo']
    
    for col in categorical_columns:
        dataset[col] = dataset[col].astype(str)
        
        if fit:
        
            all_labels = dataset[col].unique().tolist()  # extragem toate etichetele unice
            if 'unknown' not in all_labels:
                all_labels.append('unknown')  # verificam daca exista unknown si daca nu, il adaugam noi
            encoders[col] = LabelEncoder()   # initializam un Label Encoder pentru fiecare coloana
            encoders[col].fit(all_labels)
        else:
            # Inlocuim coloanele nevazute de encoder in etapa de fit cu 'unkown'
            known_classes = set(encoders[col].classes_)
            dataset[col] = dataset[col].apply(lambda x: x if x in known_classes else 'unknown')
        
        dataset[col] = encoders[col].transform(dataset[col])
    
    return dataset



################### VERIFICAM SKEWING-UL PE DATASET  ######################
def apply_log1p_if_skewed(df, threshold=1.0):
    skewed_feats = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False) # selectam coloanele numerice in ordine descrescatoare in functie de skewness
    skewed_cols = skewed_feats[skewed_feats > threshold].index

    df[skewed_cols] = np.log1p(df[skewed_cols]) # aplicam log1p pe coloanele cu skewness mare
    return df, list(skewed_cols)


################## NORMALIZAM ###################
def normalization(dataset, fit=True, scaler = None, sc=None):
    dataset = dataset.select_dtypes(include=["number"])
    if fit:
        if sc == 'Robust':
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
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


def elastic_net(X_train, y_train, X_test, y_test):
  X_train = X_train.drop(columns=['attack_cat'])

  # Use ElasticNetCV for feature selection
  elastic_net = ElasticNetCV(l1_ratio=[0.7, 0.9, 1], 
                       cv=5,
                       selection='random', 
                       tol=1e-3)
  elastic_net.fit(X_train, y_train)

  # selectam doar feature-urile care au coeficientii diferiti de 0 in urma aplicarii elastic_net
  selected_features = X_train.columns[elastic_net.coef_ != 0].tolist()
  print(f"Selected features: {selected_features}")

  # fitram ambele seturi de date incat sa folosim doar feature urile selectate
  training_set_filtered = X_train[selected_features]
  testing_set_filtered = X_test[selected_features]
  return training_set_filtered, testing_set_filtered



def change_proportion_of_data(dataset, percentage_anomalies = 0.10, total = 50000):
  anomalies = dataset[dataset['label'] == 1]
  normal_points = dataset[dataset['label'] == 0]

  total = min(len(dataset), total)
  anomalies_to_add = int(total * percentage_anomalies)
  normal_points_to_add = total - anomalies_to_add

  # extragem datele
  anomalies = anomalies.sample(n=anomalies_to_add, random_state=42)
  normal_points = normal_points.sample(n=normal_points_to_add, random_state=42)

  # construim la loc dataframe ul
  training_set_resampled = pd.concat([anomalies, normal_points])
  # balansam
  training_set_resampled = training_set_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

  return training_set_resampled


# functie de plotare a heatmap-ului pt matricea de corelatie

def plot_corr_matrix(corr_matrix):
   plt.figure(figsize=(45, 20))
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 6})
   plt.title("Matricea de corelație pentru feature-urile numerice", fontsize=20)
   plt.xticks(rotation=90, fontsize=10)
   plt.yticks(fontsize=10)

  # Ajustează marginile manual
   plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
   plt.show()

########## ANALIZAM CORELATII ######################
def corelation_matrix(training_set):

   # selectează doar coloanele numerice
   numeric_training_set = training_set.select_dtypes(include=['number'])

   # calculează matricea de corelație
   corr_matrix_train = numeric_training_set.corr().abs()

   # heatmap pt vizualizare 
   plot_corr_matrix(corr_matrix=corr_matrix_train)

  # pastram doar corelatiile de deasupra diagonalei principale
   upper_values_train = corr_matrix_train.where(np.triu(np.ones(corr_matrix_train.shape), k=1).astype(bool))
   high_corr_pairs = [
      (col, row, upper_values_train.loc[col, row])
      for col in upper_values_train.columns
      for row in upper_values_train.index
      if upper_values_train.loc[col, row] > 0.95
  ]
   columns_to_drop = set()
   for col, _, _ in high_corr_pairs:
     if col not in columns_to_drop:
       columns_to_drop.add(col)
   return list(columns_to_drop)


def load_and_preprocess_data(training_csv_path = TRAINING_CSV_PATH, 
                             testing_csv_path = TESTING_CSV_PATH,
                             target_attack_type= None,
                             rfe_n_features = 30,
                             smote_random_state = None, scaler = None):
   
    ########## INCARC DATELE #####################
    training_set = pd.read_csv(training_csv_path, header = 0)    
    testing_set = pd.read_csv(testing_csv_path, header = 0) 
    
    # OPTIONAL: FILTRARE DUPA ATAC:
    if target_attack_type:
        training_set = filter_dataset_by_attack(training_set, target_attack_type)
        testing_set = filter_dataset_by_attack(testing_set, target_attack_type)

    # curatarea setului de date
    training_set = clean_dataset(training_set)
    testing_set = clean_dataset(testing_set)

    # Feature engineering
    training_set = feature_engineering(training_set)
    testing_set = feature_engineering(testing_set)

    # encode categorical features
    training_set = preprocessing(training_set)
    testing_set = preprocessing(testing_set, fit=False)

    # obtinem coloanele care ar putea fi eliminate calculand matricea de corelatie
    columns_to_drop = corelation_matrix(training_set)

    # RENUNTAM LA COLOANELE CARE AU CORELATII MARI
    training_set = training_set.drop(columns = columns_to_drop)
    testing_set = testing_set.drop(columns = columns_to_drop)

    # separăm feature urile de labeluri
    X_train = training_set.drop(columns=['label', 'id'])
    y_train = training_set['label']
    X_test = testing_set.drop(columns=['label', 'id'])
    y_test = testing_set['label']

    # # 5. Aplicam functia de mai sus pe setul de train
    # X_train, skewed_cols = apply_log1p_if_skewed(X_train.copy()) 
    # X_test_copy = X_test.copy() 
    # X_test_copy[skewed_cols] = np.log1p(X_test_copy[skewed_cols]) # folosim skewed cols returnate de functie ca sa aplicam si in setul de test log1p pe aceleasi coloane
    # X_test = X_test_copy 

    # Normalizam
    X_train, used_scaler = normalization(X_train, fit=True, sc = scaler)
    X_test, _ = normalization(X_test, fit=False, scaler=used_scaler)

    training_set_df = training_set.copy()
    testing_set_df = testing_set.copy()

    # Reducem numărul de feature-uri folosind metoda Kbest 
    selector = SelectKBest(f_classif, k=rfe_n_features)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)  

    # Balansam clasele
    if smote_random_state:
        smote = SMOTE(random_state=smote_random_state)
        X_test, y_test = smote.fit_resample(X_test, y_test)


    return X_train, y_train, X_test, y_test, training_set_df, testing_set_df