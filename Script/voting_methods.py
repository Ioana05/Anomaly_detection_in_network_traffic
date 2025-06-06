from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from load_files import load_and_preprocess_data

X_train, y_train, X_test, y_test = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

