

import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.mse_loss(input * 100, target * 100)