
import torch
from math import exp
import torch.nn as nn
import torch.nn.functional as F
 

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        
 
    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        self.channel = channel
        self.window = self._create_window(self.window_size, self.channel).to(img1.device).type(img1.dtype)
        return self._compute_ssim(img1, img2)
    
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _compute_ssim(self, img1, img2, full=False):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 0.5
            if torch.min(img1) < 0:
                min_val = -0.5
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range
 
        padd = 0
        
        mu1 = F.conv2d(img1, self.window, padding=padd, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=padd, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
 
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=padd, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=padd, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=padd, groups=self.channel) - mu1_mu2
 
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
 
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
 
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1)
 
        if full:
            return ret, cs
        return ret


# 像素结构损失函数
class PixelStructureLoss(nn.Module):
    def __init__(self, window_size=11, alpha=0.2, beta=0.8, size_average=True, val_range=None):
        super(PixelStructureLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.ssim = SSIM(window_size=window_size, size_average=size_average, val_range=val_range)
    
    def forward(self, img1, img2):
        mse = self.mse(img1 * 100, img2 * 100) 
        ssim = self.ssim(img1, img2)
        return self.alpha * mse + self.beta * (1 - ssim ** 2)
        