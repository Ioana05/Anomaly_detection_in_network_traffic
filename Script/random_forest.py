import random

from load_files import training_set, testing_set
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import math
import numpy as np

sub_sampling_size = 256
number_of_trees = 100

# actually mi am dat seama ca nu cred ca nu este o idee buna sa folosesc one hot encoder pentru ca vor exista coloane cu o singura valoare de 1 swi restul 0,
# iar asta s ar putea sa influenteze resultatul pentru isolation forest

training_set['proto'] = LabelEncoder().fit_transform(training_set['proto'])
training_set['service'] = LabelEncoder().fit_transform(training_set['service'])
training_set['state'] = LabelEncoder().fit_transform(training_set['state'])
training_set['attack_cat'] = LabelEncoder().fit_transform(training_set['attack_cat'])


# the next step will be to also convert the ip address into an integer value
#  wait aparent nu am ip uri in dataset


#  scot prima coloana care contine id ul
anomalies = 0
for i, instance in training_set.iterrows():
    if instance['label'] == 1:
        anomalies += 1

print(anomalies)
print(len(training_set))

training_set = training_set.drop(columns = ['id'])
training_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)
#  normalizam
scaler = MinMaxScaler()
training_set = pd.DataFrame(scaler.fit_transform(training_set), columns=training_set.columns)

# cam atat cu datasetul, vezi mai tarziu daca trebuie sa normalizezi datele(s ar putea sa fie nevoie)
# print(training_set)
# so we will try to con vert the categorical values in numerical values using
# LabelEncoder which will convert the non-numerical values in integer values

# average path length is given by the estimation of average height for BST
def calculateC(n):
    # math.e ne da constanta lui Euler
    if n <= 1:
        return 0
    expected_average_path_length = 2*(math.log(n-1) + math.e) - (2*(n-1)/n)
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
    height_limit = math.ceil(sub_sampling_size)
    forest = []
    for i in range(number_of_trees):
        # sub_sampling_set = training_set.sample(sub_sampling_size)
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
    anomaly_score = 2**(-(average_path_length)/calculateC(len(training_set)))
    return anomaly_score

forest = iForest(training_set, number_of_trees, sub_sampling_size )
classified_labels = {}

for i, x in training_set.iterrows():
    result = computeAnomalyScore(x, forest)
    if result > 0.6:
        classified_labels[i] = float(1.0)
    else:
        classified_labels[i] = float(0.0)

# print(type(training_set['label'][0]))
# print(type(list(classified_labels)[0]))
print(accuracy_score(training_set['label'], list(classified_labels.values())))




