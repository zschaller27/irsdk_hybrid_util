import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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
        d = np.array(pd.read_csv(path, delimiter=',', \
            usecols=lambda x: x in features))

        # If there is no data before reset the variable
        if data is None:
            data = np.concatenate((y, d), axis=1)
        # Otherwise add to the end of the already found data
        else:
            data = np.concatenate((data, np.concatenate((y, d), axis=1)), axis=0)
    
    return data

def success_rate(pred_Y, Y):
    '''
    Calculate and return the success rate from the predicted output Y and the
    expected output.  There are several issues to deal with.  First, the pred_Y
    is non-binary, so the classification decision requires finding which column
    index in each row of the prediction has the maximum value.  This is achieved
    by using the torch.max() method, which returns both the maximum value and the
    index of the maximum value; we want the latter.  We do this along the column,
    which we indicate with the parameter 1.  Second, the once we have a 1-d vector
    giving the index of the maximum for each of the predicted and target, we just
    need to compare and count to get the number that are different.  We could do
    using the Variable objects themselves, but it is easier syntactically to do this
    using the .data Tensors for obscure PyTorch reasons.
    '''
    _,pred_Y_index = torch.max(pred_Y, 1)
    num_equal = torch.sum(pred_Y_index.data == Y.data).item()
    num_different = torch.sum(pred_Y_index.data != Y.data).item()
    rate = num_equal / float(num_equal + num_different)
    return rate 

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
    print("NN Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))
    print(confusion_matrix(y_test, y_hat, labels=range(0, 2)))
    for i in np.where(y_hat != y_test)[0]:
        print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))

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
    for i in np.where(y_hat != y_test)[0]:
        print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))

    return model

def findNeuralNetModel(x_train, y_train, x_test, y_test):
    x_train = Variable(torch.tensor(x_train, dtype=torch.float))
    y_train = Variable(torch.tensor(y_train, dtype=torch.long))
    x_test = Variable(torch.tensor(x_test, dtype=torch.float))
    y_test = Variable(torch.tensor(y_test, dtype=torch.long))

    # Make NN
    net = Net(x_train.shape[1], 2, 15)

    # Set opimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, weight_decay=0.00001)

    if torch.cuda.is_available():
        net = net.to('cuda')

    trainNN(net, x_train, y_train, x_test, y_test, 100, 16, criterion, optimizer, 1)

    y_hat = net(x_test)

    accuracy = len(np.where(y_hat == y_test)[0]) / float(len(y_test)) * 100
    print("MLP Model Accuracy: %3.3f" %accuracy)
    print("\tY Hat Max:", np.max(y_hat))
    print("\tY Hat Min:", np.min(y_hat))
    print(confusion_matrix(y_test, y_hat, labels=range(0, 2)))
    for i in np.where(y_hat != y_test)[0]:
        print("%d\ty_hat: %1.1f\tgt_y: %1.1f" %(i, y_hat[i], y_test[i]))
    
    return net

"""
Neural Network Test Model
"""
class Net(nn.Module):
    def __init__(self, input_data_features, L=2, H=15):
        super(Net, self).__init__()

        """
        Use L number of fully connected layers, each with N number of nodes each.
        The output layer will have a size of 1, with the value of the output being
        the class of that image.
        """
        self.in_layer = nn.Linear(input_data_features, H, bias=True)

        self.hidden_layers = []
        if L > 1:
            for i in range(1, L):
                self.hidden_layers.append(nn.Dropout(p=0.9))
                self.hidden_layers.append(nn.Linear(H, H, bias=True))
        
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.out_layer = nn.Linear(H, 5, bias=True)
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        x = self.out_layer(x)

        return x

def trainNN(net, x_train, y_train, x_test, y_test, epochs, batch_size, criterion, optimizer, gpu):
    if torch.cuda.is_available() and gpu:
        Y_train = y_train.to('cuda')
        X_train = x_train.to('cuda')
        Y_valid = y_test.to('cuda')
        X_valid = x_test.to('cuda')

    n_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for ep in range(epochs):
        #  Create a random permutation of the indices of the row vectors.
        indices = torch.randperm(X_train.shape[0])
        
        #  Run through each mini-batch
        for b in range(n_batches):
            #  Use slicing (of the pytorch Variable) to extract the
            #  indices and then the data instances for the next mini-batch
            batch_indices = indices[b*batch_size: (b+1)*batch_size]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train[batch_indices]

            # Check if Net and both batches are on the same device
            assert next(net.parameters()).is_cuda == batch_X.is_cuda == batch_Y.is_cuda
            
            pred_Y = net(batch_X)

            loss = criterion(pred_Y, batch_Y)
            
            #  Back-propagate the gradient through the network using the
            #  implicitly defined backward function, but zero out the
            #  gradient first.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #  Print validation loss every 10 epochs
        if ep != 0 and ep % 10 == 0:
            pred_Y_train = net(X_train)
            valid_loss = criterion(pred_Y_train, Y_train)
            print("Epoch Training %d loss: %.5f" %(ep, valid_loss.item()))

            pred_Y_valid = net(X_valid)
            valid_loss = criterion(pred_Y_valid, Y_valid)
            print("Epoch Validation %d loss: %.5f" %(ep, valid_loss.item()))

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

    # linear_model = findLinearModel(x_train, y_train, x_test, y_test)
    # print("", flush=True)
    # svc_model = findSVCModel(x_train, y_train, x_test, y_test)
    # print("", flush=True)
    # svr_model = findSVRModel(x_train, y_train, x_test, y_test)
    # print("", flush=True)
    # sgd_model = findSGDModel(x_train, y_train, x_test, y_test)
    # print("", flush=True)
    nn_model = findNNModel(x_train, y_train, x_test, y_test)
    print("", flush=True)
    # neural_network_model = findNeuralNetModel(x_train, y_train, x_test, y_test)

    # pickle.dump(nn_model, open("test_model.p", "wb"))