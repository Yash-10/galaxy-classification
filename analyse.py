import matplotlib.pyplot as plt
from classifier import *
import numpy as np

data = np.load("galaxy_catalogue.npy")
depth = best_tree_depth(data, 20)

print(f"The most optimal depth for decision tree is: {depth}")
print("\n")
predicted, actual = best_rfc(data, 50)
accuracy = calculate_accuracy(predicted, actual)
print(f"Accuracy using the most optimzed Random Forest Classifier is: {accuracy * 100} %")