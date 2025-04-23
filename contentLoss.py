
import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = models.vgg19()
        vgg.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
        for pa in vgg.parameters():
            pa.requires_grad = False
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg = vgg.features[:16]

    def forward(self, x):
        out = self.vgg(x)
        return out


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGG()

    def forward(self, fake, real):
        feature_fake = self.vgg(fake)
        feature_real = self.vgg(real)
        return self.mse(feature_fake, feature_real)










