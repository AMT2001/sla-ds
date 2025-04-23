
import torch
import numpy as np
from loss.psLoss import SSIM

# 计算MSE，支持numpy、tensor
def mse(y_true, y_pred):
    # 将最后的返回结果统一为float类型
    if isinstance(y_true, np.ndarray):
        return np.mean((y_true - y_pred) ** 2)
    elif isinstance(y_true, torch.Tensor):
        return torch.mean((y_true - y_pred) ** 2)
    else:
        raise TypeError("Unsupported data type")    
    
# 计算batch的MSE，支持numpy、tensor
def batch_mse(y_true, y_pred):
    # 根据数据类型检查数据维度是否为4
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
        
    # 将最后的返回结果统一为float类型
    result = 0
    for i in range(y_true.shape[0]):
        if y_true[i].ndim != 3 or y_pred[i].ndim != 3:
            raise ValueError("The dimension of data must be 3")
        else:
            result += mse(y_true[i], y_pred[i])
            
    return result / y_true.shape[0]

# 计算MAE，支持numpy、tensor
def mae(y_true, y_pred):
    # 将最后的返回结果统一为float类型
    if isinstance(y_true, np.ndarray):
        return np.mean(np.abs(y_true - y_pred))
    elif isinstance(y_true, torch.Tensor):
        return torch.mean(torch.abs(y_true - y_pred))
    else:
        raise TypeError("Unsupported data type")

# 计算batch的MAE，支持numpy、tensor
def batch_mae(y_true, y_pred):
    # 根据数据类型检查数据维度是否为4
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
        
    # 将最后的返回结果统一为float类型
    result = 0
    for i in range(y_true.shape[0]):
        if y_true[i].ndim != 3 or y_pred[i].ndim != 3:
            raise ValueError("The dimension of data must be 3")
        else:
            result += mae(y_true[i], y_pred[i])
            
    return result / y_true.shape[0]


# 计算RMSE，支持numpy、tensor
def rmse(y_true, y_pred):
    # 将最后的返回结果统一为float类型
    if isinstance(y_true, np.ndarray):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    elif isinstance(y_true, torch.Tensor):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    else:
        raise TypeError("Unsupported data type")

# 计算batch的RMSE，支持numpy、tensor
def batch_rmse(y_true, y_pred):
    # 根据数据类型检查数据维度是否为4
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
        
    # 将最后的返回结果统一为float类型
    result = 0
    for i in range(y_true.shape[0]):
        if y_true[i].ndim != 3 or y_pred[i].ndim != 3:
            raise ValueError("The dimension of data must be 3")
        else:
            result += rmse(y_true[i], y_pred[i])
            
    return result / y_true.shape[0]

# 计算PSNR，支持numpy、tensor, 判断最值的大小选择对应的PSNR计算方式


def psnr(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        return 20 * np.log10(np.max(y_true) / np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif isinstance(y_true, torch.Tensor):
        return 20 * torch.log10(torch.max(y_true) / torch.sqrt(torch.mean((y_true - y_pred) ** 2)))
    else:
        raise TypeError("Unsupported data type")

# 计算batch的PSNR，支持numpy、tensor
def batch_psnr(y_true, y_pred):
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
    result  = 0
    for i in range(y_true.shape[0]):
        if y_true[i].ndim != 3 or y_pred[i].ndim != 3:
            raise ValueError("The dimension of data must be 3")
        else:
            result += psnr(y_true[i], y_pred[i])
            
    return result / y_true.shape[0]


# 计算PCC，支持numpy、tensor
def pcc(y_true, y_pred):
    # 计算模式相关系数
    if isinstance(y_true, np.ndarray):
        temp = np.corrcoef(y_true.flatten(), y_pred.flatten())
        # print(temp.shape)
        return temp[0, 1]
    elif isinstance(y_true, torch.Tensor):
        return torch.corrcoef(y_true[0], y_pred[0])[0, 1]
    else:
        raise TypeError("Unsupported data type")
    

# 计算batch的PCC，支持numpy、tensor
def batch_pcc(y_true, y_pred):
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
    result  = 0
    for i in range(y_true.shape[0]):
        if y_true[i].ndim != 3 or y_pred[i].ndim != 3:
            raise ValueError("The dimension of data must be 3")
        else:
            result += pcc(y_true[i], y_pred[i])   
    return result / y_true.shape[0]

# 计算SSIM，支持numpy、tensor
def ssim(y_true, y_pred):
    ssim_loss = SSIM(window_size=3, val_range=1.0)
    return ssim_loss(y_true, y_pred)


# 计算batch的SSIM，支持numpy、tensor
def batch_ssim(y_true, y_pred):
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError("The dimension of data must be 4")
    result  = 0
    for i in range(y_true.shape[0]):
        result += ssim(y_true[i:i+1], y_pred[i:i+1])   
    return result / y_true.shape[0]

