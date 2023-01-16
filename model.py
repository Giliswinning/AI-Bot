import torch
import torch.nn as nn


class NeuralNet(nn.Module): # create new model class must be copied fromm nn.module
    def __init__(self, input_size, hidden_size, num_classes): # define init gets self and the hidden size
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

# we want to create three linear layers by saying self l1
# our second layer has hidden size as input and the hidden size also as output
# 3 layer has hidden size as input and number of classes as output
# input size and number of classes must be fixed but you can change the hidden size
# create activation function for in between we use relu activation function

    def forward(self, x): # implement forward path self and x
        out = self.l1(x) #
        out = self.relu(out) # apply activation function inbetween
        out = self.l2(out) #
        out = self.relu(out)
        out = self.l3(out)

        return out


""" This will be a feed forward neural net with two hidden layers
 """