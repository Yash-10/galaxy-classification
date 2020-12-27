from classifier import *
from knn import KNearestNeighbours
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


data = np.load("galaxy_catalogue.npy") # unbalanced data

# depth = best_tree_depth(data, 20)
#
# print(f"The most optimal depth for decision tree is: {depth}")
# print("\n")
# predicted, actual = best_rfc(data, 50)
# accuracy = calculate_accuracy(predicted, actual)
# print(f"Accuracy using the most optimzed Random Forest Classifier is: {accuracy * 100} %")
#def analyze(data, test_frac=0.1):


test_frac = 0.1
num_test = int(len(data) * test_frac)
random.shuffle(data)
data = pd.DataFrame(data)

print(f"No. of merger galaxies: {data[data['class'] == 'merger'].shape[0]}")
print(f"No. of elliptical galaxies: {data[data['class'] == 'elliptical'].shape[0]}")
print(f"No. of spiral galaxies: {data[data['class'] == 'spiral'].shape[0]}")

feed_data = np.array(data.loc[:, data.columns != 'class'])
labels = np.array(data.loc[:, data.columns == 'class'])

X_test = feed_data[:num_test]
X_train = feed_data[num_test:]
y_test = labels[:num_test]
y_train = labels[num_test:]

knn_classifier = KNearestNeighbours()
knn_classifier.train(X_train, y_train)
dists = knn_classifier.dist_calc(X_test)
pred_labels = knn_classifier.predict_labels(dists)

num_correct = np.sum(pred_labels == y_test)
accuracy = float(num_correct) / num_test
print(f"Test Accuracy: {accuracy}")
