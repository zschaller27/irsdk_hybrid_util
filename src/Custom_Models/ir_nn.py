import torch
import time

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class iRacing_Network(nn.Module):
    def __init__(self, num_features, num_layers=2, nodes_per_layer=15):
        super(iRacing_Network, self).__init__()

        # Build Network Architecture
        self.in_layer = nn.Linear(num_features, nodes_per_layer, bias=True)
        self.output_layer = nn.Linear(nodes_per_layer, 2, bias=True)        # Use 1-hot classification

        self.hidden_layers = []
        if num_layers > 1:
            for _ in range(num_layers - 1):
                # self.hidden_layers.append(nn.Dropout(p=0.9))   # To help avoid overfitting
                self.hidden_layers.append(nn.Linear(nodes_per_layer, nodes_per_layer, bias=True))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
    
    def forward(self, data):      
        x = F.relu(self.in_layer(data))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)
    
def trainNetwork(network, x_train, y_train, optimizer, epochs=10, batch_size=16, loss_function=nn.CrossEntropyLoss()):
    # Make sure data is stored in torch Variables
    if not isinstance(x_train, Variable):
        x_train = Variable(torch.tensor(x_train, dtype=torch.float))
    if not isinstance(y_train, Variable):
        y_train = Variable(torch.tensor(y_train, dtype=torch.long))

    # Check if cuda is avalible
    if torch.cuda.is_available():
        print("Using GPU")
        x_train = x_train.to('cuda')
        y_train = y_train.to('cuda')
        network = network.to('cuda')

    # Determine the number of batches (if batch_size == -1 then there should be 1 batch)
    if batch_size == -1:
        num_batches = 1
    else:
        num_batches = int(np.ceil(x_train.shape[0] / batch_size))
    
    prev_loss = None

    for epoch in range(epochs):
        epoch_start = time.time()

        # Shuffle a list of avalible indicies (randomizes batches)
        indicies = torch.randperm(x_train.shape[0])

        # Train on each batch
        for batch in range(num_batches):
            batch_indicies = indicies[batch * batch_size : (batch + 1) * batch_size]
            x_batch = x_train[batch_indicies]
            y_batch = y_train[batch_indicies]

            # Predict for batch
            y_hat = network(x_batch)

            # Find loss
            loss = loss_function(y_hat, y_batch)

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        #  Print validation loss every 10 epochs
        if epoch % 10 == 0:
            print("Epoch: %d\tloss: %.5f\tDuration: %3.3f" %(epoch, loss.item(), time.time() - epoch_start), flush=True)
        
        # Check for convergence
        if prev_loss is not None and (np.abs(loss - prev_loss) < 0.01 or loss - prev_loss > 0.1):
            print("Convergence Detected")
            break

    # Move the network back to cpu for future use
    network = network.to('cpu')

def success_rate(net, x, y):
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
    # Make sure data is stored in torch Variables
    if not isinstance(x, Variable):
        x = Variable(torch.tensor(x, dtype=torch.float))
    if not isinstance(y, Variable):
        y = Variable(torch.tensor(y, dtype=torch.long))

    pred_Y = net(x)

    _, pred_Y_index = torch.max(pred_Y, 1)
    
    num_equal = torch.sum(pred_Y_index.data == y.data).item()
    num_different = torch.sum(pred_Y_index.data != y.data).item()
    rate = num_equal / float(num_equal + num_different)

    return pred_Y_index.detach().numpy(), rate
