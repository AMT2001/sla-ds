# This file contains the implementation of Mean Absolute Error Loss function.
import torch
from torch.nn import Module

class MAELoss(Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input * 100, target * 100)