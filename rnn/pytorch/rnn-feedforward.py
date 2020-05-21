import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleRNN(nn.Module):
    """
    Simple RNN example where everything is hardcoded
    """
    def __init__(self, n_inputs, n_neurons):
        super(SimpleRNN, self).__init__()
        # w1: weight for input x_n
        # w2: weight for previous result y_{n-1}
        self.w1 = torch.randn(n_inputs, n_neurons)
        self.w2 = torch.randn(n_neurons, n_neurons)
        
        # bias term
        self.b = torch.zeros(1, n_neurons)

    def forward(self, x0, x1):
        # hand compute the two time-steps
        self.y0 = torch.tanh(torch.mm(x0, self.w1) + self.b)
        self.y1 = torch.tanh(torch.mm(self.y0, self.w2) +
                             torch.mm(x1, self.w1) + self.b)
        return self.y0, self.y1

class BasicRNN(nn.Module):
    """
    Basic RNN that uses pytorch's rnn cell
    """
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(BasicRNN, self).__init__()
        # rnn
        ## RNNCell (takes care of creating and maintaining weights and biases
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        ## Intermediary states after each timestep that will be used as supplementary input
        ##  to the RNNCell (y_{n-1})
        self.hx = torch.randn(batch_size, n_neurons)
    
    def forward(self, x):
        output = []
        # automate all time-steps
        for i in range(2):
            self.hx = self.rnn(x[i], self.hx)
            output.append(self.hx)

        # all outputs, last output
        return output, self.hx

def main():
    # Simple RNN Example
    print("Simple RNN feed-forward example")
    print()
    n_input = 4
    n_neurons = 1

    x0_batch = torch.tensor([[0,1,2,3], 
                             [2,3,4,5],
                             [3,4,5,6],
                             [4,5,6,7]],
                             dtype=torch.float)
    x1_batch = torch.tensor([[9,8,7,6],
                             [8,7,6,5],
                             [7,6,5,4],
                             [6,5,4,3]],
                             dtype=torch.float)

    model = SimpleRNN(n_input, n_neurons)
    
    # Feed-forward 
    y0_pred, y1_pred = model(x0_batch, x1_batch)
    print(y0_pred)
    print(y1_pred)

    # Basic RNN Example
    print("Basic RNN feed-forward example")
    print()
    batch_size = 4
    n_input = 3
    n_neurons = 5
    x_batch = torch.tensor([[[0,1,2], [1,2,3], [2,3,4], [3,4,5]],  # x0
                            [[9,8,7], [8,7,6], [7,6,5], [6,5,4]]], # x1
                           dtype=torch.float)
    model = BasicRNN(batch_size, n_input, n_neurons)
    outputs, result = model(x_batch)
    print(outputs)
    print(result)

if __name__ == '__main__':
    main()
