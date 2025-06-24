import random
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import math
import numpy as np
import matplotlib.pyplot as plt
from Script.pipeline import  load_and_preprocess_data,  change_proportion_of_data

X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_data(target_attack_type= None, rfe_n_features=30 )

training_set_resampled = pd.DataFrame(X_train)
training_set_resampled['label'] = y_train

testing_set_resampled = pd.DataFrame(X_test)
testing_set_resampled['label'] = y_test

train_anomalies = 0.1
# Schimbam rata de infectare doar pe setul de antrenare
training_set_resampled = change_proportion_of_data(training_set_resampled, percentage_anomalies=train_anomalies)

# Dupa ce am schimbat rata, separam iarasi datele de antrenare de datele de test
X_train = training_set_resampled.drop(columns=['label'])  
y_train = training_set_resampled['label']                 
        
# lungimea medie a drumurilor este inaltimea medie calculata folosind formulele BST
def calculateC(n):
    gamma = 0.5772156649
    if n <= 1:
        return 0
    expected_average_path_length = 2.0*(math.log(n-1) + gamma) - (2.0*(n-1)/(n*1.0))
    return expected_average_path_length

def iTree(subset, current_tree_height, limit_of_height):
    if current_tree_height >= limit_of_height or len(subset) <= 1:
        return {"Size": len(subset)}
    else:
        # includem doar feature-urile numerice
        numeric_columns = subset.select_dtypes(include=np.number).columns.tolist()

        # ne asiguram ca au fost gasite coloane numerice
        if not numeric_columns:
             return {"Size": len(subset)} 

        # selectam un atribut random in functie de care sa se faca splitul
        random_attr = random.choice(numeric_columns)

        # gasim valoarea minima si maxima pentru a putea forma un interval din care sa selectam valoarea de split
        min_value = subset[random_attr].min()
        max_value = subset[random_attr].max()

        # luam in considerare cazul in care min si max sunt egale
        if min_value == max_value:
            return {"Size": len(subset)}

        # alegem valoarea de split
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

# construim ansamblul de arbori 
def iForest(training_set, number_of_trees, sub_sampling_size):
    height_limit = math.ceil(math.log(sub_sampling_size))
    forest = []
    for i in range(number_of_trees):
        sub_sampling_set = training_set.sample(sub_sampling_size).reset_index(drop = True)
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

# scorul de anomalie va fi calculat conform formulei din paper
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
def run_isolation_forest():

    for epoch in range(len(number_of_trees)):

        forest = iForest(training_set_resampled, number_of_trees[epoch], sub_sampling_size[epoch])
        classified_labels = {}
        scores = []

        for i, x in testing_set_resampled.iterrows():
            result = computeAnomalyScore(x, forest)
            scores.append(result)

        for i, x in testing_set_resampled.iterrows():
            # 0.43 este cel mai bun threshold gasit
            if scores[i] > 0.43:
                classified_labels[i] = float(1.0)
            else:
                classified_labels[i] = float(0.0)

        print(f"Number of trees: {number_of_trees[epoch]} and sub-sampling size: {sub_sampling_size}")
        print(accuracy_score(testing_set_resampled['label'], list(classified_labels.values())))
        print(classification_report(testing_set_resampled['label'], list(classified_labels.values())))
        print(roc_auc_score(testing_set_resampled['label'], list(classified_labels.values())))

        threshold = np.percentile(scores, 50)
        print(threshold)
        for i, x in testing_set_resampled.iterrows():
            if scores[i] > threshold:
                classified_labels[i] = float(1.0)
            else:
                classified_labels[i] = float(0.0)

        print(f"Number of trees: {number_of_trees[epoch]} and sub-sampling size: {sub_sampling_size}")
        print(accuracy_score(testing_set_resampled['label'], list(classified_labels.values())))
        print(roc_auc_score(testing_set_resampled['label'], list(classified_labels.values())))
        print(classification_report(testing_set_resampled['label'], list(classified_labels.values())))



    anomalies = [score for score, is_anomaly in zip(scores, testing_set_resampled['label']) if is_anomaly]
    normal_points = [score for score, is_anomaly in zip(scores, testing_set_resampled['label']) if not is_anomaly]
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


run_isolation_forest()