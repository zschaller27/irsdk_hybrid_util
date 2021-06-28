# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# Data
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn

import torch
import numpy as np
import pandas as pd
import Custom_Models.ir_nn as iRacing_NN

import pickle
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
        d = np.array(pd.read_csv(path, delimiter=',', usecols=lambda x: x in features))

        # If there is no data before reset the variable
        if data is None:
            data = np.concatenate((y, d), axis=1)
        # Otherwise add to the end of the already found data
        else:
            data = np.concatenate((data, np.concatenate((y, d), axis=1)), axis=0)

    return data

def balanceDataSet(dataset):
    """
    Functiuon to balance the dataset between the two possible classes. This will improve the 
    training performance of the model.

    Parameters:
        dataset : dataset to balance
    
    Return:
        If there is more of one class in the dataset, return a new dataset where the two classes
        are equal. Otherwise return the original dataset.
    """
    class_0_indicies = np.array(np.where(dataset[:, 0] == 0)[0])
    class_1_indicies = np.array(np.where(dataset[:, 0] == 1)[0])

    if class_0_indicies.shape[0] > class_1_indicies.shape[0]:
        np.random.shuffle(class_0_indicies)
        choosen_0_indicies = np.random.choice(class_0_indicies, size=class_1_indicies.shape[0])
        new_dataset = np.concatenate((dataset[class_1_indicies], dataset[choosen_0_indicies]), axis=0)
        return new_dataset
    
    return dataset
    
"""
The following functions will return a desired model by either loading it from storage
or training a new one if one isn't trained in the past.
"""

def getNearestNeighborModel(features, path="D:/Personal Projects/irsdk_hybrid_util/Data/Audi/"):
    if os.path.exists("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/nearest_neighbor_model.p"):
        return pickle.load(open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/nearest_neighbor_model.p", 'rb'))
    
    data = loadData(findPaths(path), features) 
    x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.1, random_state=0)
    return findNNModel(x_train, x_test, y_train, y_test)

def getNeuralNetworkModel(features, path="D:/Personal Projects/irsdk_hybrid_util/Data/Audi/"):
    if os.path.exists("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/neural_network_model.p.p"):
        return pickle.load(open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/neural_network_model.p.p", 'rb'))
    
    data = loadData(findPaths(path), features) 
    x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.1, random_state=0)
    return findNNModel(x_train, x_test, y_train, y_test)

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

def findNNModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = KNeighborsClassifier(n_neighbors=np.sqrt(x_train.shape[0]).astype(int), weights='distance')

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("Nearest Neighbor Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))
    print(confusion_matrix(y_test, y_hat, labels=range(0, 2)))
    # for i in np.where(y_hat != y_test)[0]:
    #     print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))

    return model

def findMLPModel(x_train, y_train, x_test, y_test):
    # Create linear regression model
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_hat = np.round(model.predict(x_test))

    # Print accuracy
    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("MLP Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))
    print(confusion_matrix(y_test, y_hat, labels=range(0, 2)))
    # for i in np.where(y_hat != y_test)[0]:
    #     print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))

    return model

def findNeuralNetModel(x_train, y_train, x_test, y_test):
    # Make NN
    net = iRacing_NN.iRacing_Network(x_train.shape[1], 20, 50)

    # Set opimizer and criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.999)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    ## Test Code ##
    print("Neural Net: Starting Training", flush=True)
    ## ENd Test Code ##

    iRacing_NN.trainNetwork(net, x_train, y_train, optimizer, 100000, -1, criterion)

    y_hat, accuracy = iRacing_NN.success_rate(net, x_test, y_test)

    print("Neural Net Model Accuracy: %3.3f" %(accuracy * 100))
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))
    print(confusion_matrix(y_test, y_hat, labels=range(0, 2)))
    # for i in np.where(y_hat != y_test)[0]:
    #     print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))
    
    return net

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
        "SteeringWheelAngle", "Throttle", "VelocityY", "dcMGUKDeployFixed", "dcMGUKDeployMode", "dcMGUKRegenGain"]

    data = loadData(findPaths(data_dir), features)

    data = balanceDataSet(data)

    x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.1, random_state=0)

    ##
    # Below prints out the performance of the various models on a given data set.
    # It looks like nearest neighbor works best (not suprising), but the storage complexity
    # is a concern.
    ##

    # linear_model = findLinearModel(x_train, y_train, x_test, y_test)
    # pickle.dump(linear_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/linear_model.p", "wb"))
    # print("", flush=True)

    # svc_model = findSVCModel(x_train, y_train, x_test, y_test)
    # pickle.dump(svc_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/svc_model.p", "wb"))
    # print("", flush=True)

    # svr_model = findSVRModel(x_train, y_train, x_test, y_test)
    # pickle.dump(svr_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/svr_model.p", "wb"))
    # print("", flush=True)

    # sgd_model = findSGDModel(x_train, y_train, x_test, y_test)
    # pickle.dump(sgd_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/sgd_model.p", "wb"))
    # print("", flush=True)

    # nn_model = findNNModel(x_train, y_train, x_test, y_test)
    # pickle.dump(nn_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/nearest_neighbor_model.p", "wb"))
    # print("", flush=True)

    # mlp_model = findMLPModel(x_train, y_train, x_test, y_test)
    # pickle.dump(mlp_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/mlp_model.p", "wb"))
    # print("", flush=True)

    net_model = findNeuralNetModel(x_train, y_train, x_test, y_test)
    pickle.dump(net_model, open("D:/Personal Projects/irsdk_hybrid_util/src/Generated_Models/neural_network_model.p", "wb"))
