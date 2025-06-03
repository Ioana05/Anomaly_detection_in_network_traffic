import random

from load_files import training_set, testing_set
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from load_files import X_res, y_res, X_test, y_test, testing_set

# actually mi am dat seama ca nu cred ca nu este o idee buna sa folosesc one hot encoder pentru ca vor exista coloane cu o singura valoare de 1 swi restul 0,
# iar asta s ar putea sa influenteze resultatul pentru isolation forest

training_set_resampled = pd.DataFrame(X_res)
training_set_resampled['label'] = y_res


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


def run_isolation_forest():
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

