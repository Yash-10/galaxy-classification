from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = np.load("galaxy_catalogue.npy")

df = pd.DataFrame(data) # Printing 'df' will give a table format to data

def split_data(data, train_frac):
    """
    Returns training and testing set based on the training fraction
    """
    np.random.seed(0) # Ensure results are same when trying to run this function
    np.random.shuffle(data) # Shuffle the data for smooth division of dataset
    split_index = int(len(data) * train_frac)
    train_set = data[:split_index]
    test_set = data[split_index:]

    return train_set, test_set

def generate_features_targets(data):
    """
    Generates features and targets from original data:
    Features: Color indices (u-g, g-r, r-i, i-z), Eccentricity (median across the 5 filters), 4th Adaptive moments and the concentration (petro50 / petro90)
    Target: Class (Spiral or Merger or Elliptical)
    """
    targets = data["class"]
    features = np.empty(shape=(len(data), 13))
    features[:, 0] = data["u-g"]
    features[:, 1] = data["g-r"]
    features[:, 2] = data["r-i"]
    features[:, 3] = data["i-z"]
    features[:, 4] = data["ecc"]
    features[:, 5] = data["m4_u"]
    features[:, 6] = data["m4_g"]
    features[:, 7] = data["m4_r"]
    features[:, 8] = data["m4_i"]
    features[:, 9] = data["m4_z"]
    features[:, 10] = data["petroR50_u"] / data["petroR90_u"]
    features[:, 11] = data["petroR50_r"] / data["petroR90_r"]
    features[:, 12] = data["petroR50_z"] / data["petroR90_z"]

    return features, targets

def dtc_predict_actual(data):
    """
    Predicts type of galaxy using a decision tree classifier
    """
    train_data, test_data = split_data(data, 0.7)
    train_features, train_targets = generate_features_targets(train_data)
    test_features, test_targets = generate_features_targets(test_data)
 
    dtc = DecisionTreeClassifier() # Instantiate a decision tree classifier
    dtc.fit(train_features, train_targets) # Fit the decision tree

    predictions = dtc.predict(test_features) # Predict

    return predictions, test_targets

def calculate_accuracy(predicted, actual):
    return sum(predicted == actual) / len(predicted)

###############MAKE THIS WORK##############
def best_tree_depth(data, limit):
    """
    Calculates the best tree depth that leads to the best accuracy in classification
    limit: The maximum tree depth to be taken in the analysis - anything between 10-20 or even lesser may be good.
    """
    

    #return optim_tree_depth
###########################################

###############make this work correctly#########################
def dtc_predict_actual_refined(data, limit):
    """
    # Calculates the best tree depth that leads to the best accuracy in classification, limit: The maximum tree depth to be taken in the analysis - anything between 10-20 is good.
    """
    accuracies = []
    acc = -math.inf
    print(f"This is accuracies: {accuracies}")
    for i in range(1, limit):
        train_data, test_data = split_data(data, 0.7)
        train_features, train_targets = generate_features_targets(train_data)
        test_features, test_targets = generate_features_targets(test_data)
        dtc = DecisionTreeClassifier(max_depth=i) # In each iteration, tree depth is incremented
        dtc.fit(train_features, train_targets) 
        predictions = dtc.predict(test_features)
        accuracy = calculate_accuracy(predictions, test_targets)
        if accuracy > acc:
            acc = accuracy
            pred = predictions
        accuracies.append(accuracy)
    optim_tree_depth = accuracies.index(max(accuracies)) + 1

    return pred, test_targets 
###############################################################

def dtc_predict_actual_best(data, k):
    """
    Validation using K-Fold cross validation - No need of training and testing sets
    """
    features, targets = generate_features_targets(data)
    dtc = DecisionTreeClassifier()
    predicted = cross_val_predict(dtc, features, targets, cv=10) # cv is the value of K (no of subsets to make of the whole data)
    
    #If need model_score for each fold, use cross_val_score and take average of each.
    ###### model_score = cross_val_score(dtc, features, targets, cv=10) ######
    
    #model_score = calculate_accuracy(predicted, targets) --> To get the accuracy

    return predicted, targets

def rfc_predict(data, k):
    """
    Random Forest Classifier (ensemble of decision trees) with K-Fold cross validation - Again, no need of training and testing sets
    """
    features, targets = generate_features_targets(data)
    rfc = RandomForestClassifier(n_estimators=k)
    predictions = cross_val_predict(rfc, features, targets, cv=10)

    return predictions, targets

def best_rfc(data, k):
    """
    Finds the most optimized random forest classifer (checks till number of estimators = k)
    """
    _, targets = generate_features_targets(data)
    acc = -math.inf
    for i in range(k):
        predicted, actual = rfc_predict(data, k)
        accuracy = calculate_accuracy(predicted, actual)
        if accuracy > acc:
            acc = accuracy
            pred = predicted
    return pred, targets
