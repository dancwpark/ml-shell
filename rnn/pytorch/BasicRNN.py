import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
