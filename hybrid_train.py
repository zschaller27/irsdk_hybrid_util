import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split

import sys
import os

"""
This file holds the functions used to train different types of classification
models to see which one fits the data best. I made use of sklearn's wide array
of prebuilt models since they are very fast and simple to use in implementation.

I've found that the SVC model has the best results after initial testing.
"""

def findPaths(directory):
    """
    Find paths of all .csv files in the given directory.

    Paramters:
        directory : the directory to search for comma separated files.
    
    Return:
        A list of string objects which are the paths of each data file.
    """
    # Make sure the path has a '/' at the end
    if directory[-1] != '/':
        directory += '/'
    
    # List to hold all the paths found
    data_paths = []

    # Find all possible dir or files in the given path
    _, dirnames, filenames = next(os.walk(directory))
    
    # If there are any files found add them to the data_paths list (must have .jpeg file type)
    if len(filenames) > 0:
        [data_paths.append(directory + f) for f in filenames if ".csv" in f.lower()]

    # Check any dir found for images
    if len(dirnames) > 0:
        [data_paths.extend(findPaths(directory + p)) for p in dirnames]
    
    # Return image paths found
    return data_paths

def loadData(paths, features):
    """
    Take a given set of paths and read in the given features and ground truth labels into
    a single dataset. The first column of the dataset is the ground truth label for each
    data point.

    Paramters:
        paths : the list of paths to read in data from.
        features : a list of strings which are the column headers to use in classification.
    
    Return:
        A numpy matrix where each row is a data point and the columns are:
                 Index 0            Indecies 1 - len(features)
            ground truth label              features
    """
    # Make variable to hold all the data imported
    data = None
    
    # Go through each path given and add the found data
    for path in paths:
        # Find the classification (on boost or not)
        y = np.array(pd.read_csv(path, delimiter=',', usecols=lambda x: x in ["ManualBoost"]))

        # Find the features
        d = np.array(pd.read_csv(path, delimiter=',', \
            usecols=lambda x: x in features))

        # If there is no data before reset the variable
        if data is None:
            data = np.concatenate((y, d), axis=1)
        # Otherwise add to the end of the already found data
        else:
            data = np.concatenate((data, np.concatenate((y, d), axis=1)), axis=0)
    
    return data

"""
The following functions train and return different types of models. They all also
print out the accuracy of the trained model on the test dataset.
"""

def findLinearModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("Linear Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))

    return model

def findSVCModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = svm.SVC()

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("SVC Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))

    return model

def findSVRModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = svm.SVR()

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("SVR Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))

    return model

def findSGDModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = SGDClassifier(loss="hinge", penalty='l2', max_iter=1000)

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("SGD Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))

    return model
   
if __name__ == "__main__":
    # Check cmd line arguments
    if len(sys.argv) != 2:
        print("ERROR: invalid number of arguments given.\nExpected: 1\tGot: %d" %(len(sys.argv) - 1))
        sys.exit()
    # Check if given arg is a dirctory
    elif not os.path.isdir(sys.argv[1]):
        print("ERROR: path given is not a directory. Got: %s" %sys.argv[1])
        sys.exit()
    
    # Save the given dirctory
    data_dir = sys.argv[1]

    # Hard coded features
    features = ["Brake", "EnergyERSBatteryPct", "EnergyMGU_KLapDeployPct", "Speed", \
        "SteeringWheelAngle", "Throttle", "VelocityY", "SteeringWheelAngle", \
        "dcMGUKDeployFixed", "dcMGUKDeployMode", "dcMGUKRegenGain"]

    data = loadData(findPaths(data_dir), features)

    x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.2, random_state=0)

    linear_model = findLinearModel(x_train, y_train, x_test, y_test)
    print("", flush=True)
    svc_model = findSVCModel(x_train, y_train, x_test, y_test)
    print("", flush=True)
    svr_model = findSVRModel(x_train, y_train, x_test, y_test)
    print("", flush=True)
    sgd_model = findSGDModel(x_train, y_train, x_test, y_test)
    print("", flush=True)