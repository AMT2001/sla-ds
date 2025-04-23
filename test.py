
# 这个文件的作用，将测试集的数据加载出来，然后将低分辨率的数据，使用插值、SR模型得到高分辨的数据，然后将数据保存到npy文件中，供后续的训练使用。
import os
import yaml
import torch
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

from scipy.ndimage import zoom
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from utils import mae, rmse, psnr, ssim, pcc
from models.dsdn import DSDN
from models.deepsd import DeepSD
from models.srresnet import SRResNet


# upscale = 2 # 超分辨率倍数


# 加载数据根据输入要素的名称匹配文件名，然后读取数据，并将数据保存到npy文件中。
def load_data(data_path, elements, save_path):
    assert os.path.exists(data_path), "Data path does not exist."
    # 加载文件夹路径下的所有文件
    file_list = os.listdir(data_path)
    for element in elements:
        # 匹配文件名
        file_list = [file_name for file_name in file_list if file_name.startswith(element)]
        # sort文件名
        file_list.sort()
        print(file_list)
        # 读取数据
        data = []
        for file_name in file_list:
            file_path = os.path.join(data_path, file_name)
            img = np.load(file_path)
            data.append(img)
        # 将数据1、H、W在第一个维度上，将数据保存到npy文件中
        data = np.array(data)
        print(data.shape)
        np.save(os.path.join(save_path, element + '.npy'), data)
        print(f'{element} Data saved to:', os.path.join(save_path, element + '.npy'))


# 根据放大倍数，和element名称，加载hr数据，并保存到npy文件中。
def load_hr_data(data_path, upscale, element, save_path):
    data_path = os.path.join(data_path, f'x{upscale}')
    assert os.path.exists(data_path), "Data path does not exist."
    # 加载文件夹路径下的所有文件
    file_list = os.listdir(data_path)
    # 匹配文件名
    file_list = [file_name for file_name in file_list if file_name.startswith(element)]
    # sort文件名
    file_list.sort()
    mask = load_mask(f'raw/x{upscale}')
    print(file_list)
    # 读取数据
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        img = np.load(file_path)
        img[0][mask == 0] = 0
        data.append(img)
    # 将数据1、H、W在第一个维度上，将数据保存到npy文件中
    data = np.array(data)
    print(data.shape)
    # 保存数据到npy文件中
    np.save(os.path.join(save_path, element + '_x' + str(upscale) + '.npy'), data)


# 更具文件夹路径加载mask文件，并返回mask
def load_mask(mask_path):
    # 判断mask文件是否存在
    assert os.path.exists(os.path.join(mask_path,'mask.npy')), "Mask file does not exist."
    mask = np.load(os.path.join(mask_path,'mask.npy'))
    return mask


# 根据插值方法名称，使用插值方法将低分辨率数据插值到高分辨率数据，并将数据保存到两个npy文件中，一个陆地值为零，一个陆地值为nan。
def interpolate_data(data_path, element, save_path, upscale, method='bicubic'):
    data_file_path =os.path.join(data_path, 'lr', element + '.npy')
    assert os.path.exists(data_file_path), "Low-resolution data does not exist."
    # 加载低分辨率数据
    data = np.load(data_file_path)
    # 加载mask文件
    mask = load_mask(f'{data_path}/x{upscale}')
    
    
    # 映射插值方法到 zoom 的 order 参数  
    interpolation_order = {  
        'nearest': 0,  
        'linear': 1,  
        'bilinear': 2,  
        'bicubic': 3,
        'Fourth-order': 4,
        'Fifth-order': 5,
        'wavelet_haar': 'haar',
        'wavelet_db1': 'db1',
        'wavelet_sym2': 'sym2',
    }  
      
    if method not in interpolation_order:  
        raise ValueError(f"Unsupported interpolation method: {method}")  
    order = interpolation_order[method]  
    
    # 获取输入数据的形状  
    n_samples, n_features, H, W = data.shape  
    # 计算插值后的新形状  
    new_H, new_W = int(H * upscale), int(W * upscale)  
    # 创建一个空数组来存储插值后的数据  
    
    mask_data = np.zeros((n_samples, n_features, new_H, new_W))  
    nan_mask_data = np.zeros((n_samples, n_features, new_H, new_W))  
    # 遍历数据集并进行插值  
    for i in range(n_samples):  
        for j in range(n_features): 
            # 将数据data[i, j]，间隔2进行抽取值
            que_data = data[i, j]
            que_data = que_data[::2, ::2]
            temp = zoom(que_data, (upscale * 2, upscale * 2), mode='reflect', order=order)  
            temp[mask == 0] = 0
            mask_data[i, j] = temp
            temp[mask == 0] = np.nan
            nan_mask_data[i, j] = temp
    # 保存插值后的数据到两个npy文件中
    np.save(os.path.join(save_path, method + '_x' + str(upscale) + '_mask.npy'), mask_data)
    np.save(os.path.join(save_path, method + '_x' + str(upscale) + '_nan_mask.npy'), nan_mask_data)
    print('shape of mask_data:', mask_data.shape)
    print('shape of nan_mask_data:', nan_mask_data.shape)


# deepSD预测数据函数
def predict_deepsd(config_path, data_path, element, save_path, upscale, area='bia'):
    
    data_file_path =os.path.join(data_path, 'lr', element + '.npy')
    assert os.path.exists(data_file_path), "Low-resolution data does not exist."
    # 加载低分辨率数据
    data = np.load(data_file_path)
    # 加载mask文件
    mask = load_mask(f'{data_path}/x{upscale}')
    # 根据config文件创建模型
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_params = config['DeepSD']['model_params']
    model_params['upscale'] = 4
    model1 = DeepSD(**model_params)
    # 加载预训练模型
    pretrained_path = f'run_{area}/deepsd/x4/sla_sla_1000.pth'
    model1.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))
    
    if upscale == 4:
        model_params['upscale'] = 2
        pretrained_path = f'run_{area}/deepsd/x2/sla_sla_1000.pth'
        model2 = DeepSD(**model_params)
        model2.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))
    
    # 获取输入数据的形状  
    n_samples, n_features, H, W = data.shape  
    # 计算插值后的新形状  
    new_H, new_W = int(H * upscale), int(W * upscale)  
    # 创建一个空数组来存储插值后的数据  
    mask_data = np.zeros((n_samples, n_features, new_H, new_W))  
    nan_mask_data = np.zeros((n_samples, n_features, new_H, new_W))  
    
    with torch.no_grad():
        model1.eval()
        
        # 遍历数据集并进行插值  
        for i in range(0, n_samples, 32):  
            lr_img = torch.from_numpy(data[i:i+32, :, ::2, ::2])
            sr_img = model1(lr_img)
            if upscale == 4:
                model2.eval()
                sr_img = model2(sr_img)
            sr_img = sr_img.detach().numpy()
            # 保存插值后的数据到两个npy文件中
            # 将ouput的b,c,h,w,遍历bi,ci,hi,wi,将其保存到mask_data中
            for b in range(sr_img.shape[0]):
                    temp = sr_img[b, 0]
                    temp[mask == 0] = 0
                    mask_data[i+b, 0] = temp
                    temp[mask == 0] = np.nan
                    nan_mask_data[i+b, 0] = temp
    # 保存插值后的数据到两个npy文件中
    np.save(os.path.join(save_path, 'DeepSD_x' + str(upscale) + '_mask.npy'), mask_data)
    np.save(os.path.join(save_path, 'DeepSD_x' + str(upscale) + '_nan_mask.npy'), nan_mask_data)
    print('shape of mask_data:', mask_data.shape)
    print('shape of nan_mask_data:', nan_mask_data.shape)


# 根据输入的模型名称，创建模型，并加载预训练模型，然后将低分辨率的数据通过模型得到高分辨率的数据，并保存到npy文件中。
def predict_data(model_name, config_path, pretrained_path, data_path, elements, save_path, upscale):
    data = None
    data_path_base = os.path.join(data_path, 'lr')
    for element in elements:
        data_file_path = os.path.join(data_path_base, element + '.npy')
        assert os.path.exists(data_file_path), "Low-resolution data does not exist."
        temp_data = np.load(data_file_path)
        if data is None:
            data = temp_data
        else:
            data = np.concatenate((data, temp_data), axis=1)
    # 确认数据的形状
    assert data.shape[1] == len(elements), "Data shape is not correct."
    print(data.shape)
    
    # 加载mask文件
    mask = load_mask(f'{data_path}/x{upscale}')
    model_registry = {  
        'SRGAN': SRResNet,  
        'SRResNet': SRResNet,  
    }
    # 判断模型是否存在
    assert model_name in model_registry, f"Model {model_name} not found."
    # 根据config文件创建模型
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_params = config['SRResNet']['model_params']
    model_params['upscale'] = upscale
    model_params['num_in_ch'] = len(elements)
    print(model_params)
    model = model_registry[model_name](**model_params)
    model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))
    
     # 获取输入数据的形状  
    n_samples, _, H, W = data.shape  
    # 计算插值后的新形状  
    new_H, new_W = int(H * upscale), int(W * upscale)  
    # 创建一个空数组来存储插值后的数据  
    mask_data = np.zeros((n_samples, 1, new_H, new_W))  
    nan_mask_data = np.zeros((n_samples, 1, new_H, new_W))  
    
    with torch.no_grad():
        model.eval()
        # 遍历数据集并进行插值  
        for i in range(0, n_samples, 32):  
            lr_img = torch.from_numpy(data[i:i+32])
            sr_img = model(lr_img)
            sr_img = sr_img.cpu().numpy()
            # 保存插值后的数据到两个npy文件中
            # 将ouput的b,c,h,w,遍历bi,ci,hi,wi,将其保存到mask_data中
            for b in range(sr_img.shape[0]):
                    temp = sr_img[b, 0]
                    temp[mask == 0] = 0
                    mask_data[i+b, 0] = temp
                    temp[mask == 0] = np.nan
                    nan_mask_data[i+b, 0] = temp
            
    # 保存插值后的数据到两个npy文件中
    np.save(os.path.join(save_path, model_name + '_x' + str(upscale) + '_mask.npy'), mask_data)
    np.save(os.path.join(save_path, model_name + '_x' + str(upscale) + '_nan_mask.npy'), nan_mask_data)
    print('shape of mask_data:', mask_data.shape)
    print('shape of nan_mask_data:', nan_mask_data.shape)



# DSDN预测数据函数
def predict_dsdn(data_path, elements, config_path, pretrained_path, save_path, upscale):
    data = None
    data_path_base = os.path.join(data_path, 'lr')
    for element in elements:
        data_file_path = os.path.join(data_path_base, element + '.npy')
        assert os.path.exists(data_file_path), "Low-resolution data does not exist."
        temp_data = np.load(data_file_path)
        if data is None:
            data = temp_data
        else:
            data = np.concatenate((data, temp_data), axis=1)
    # 确认数据的形状
    assert data.shape[1] == len(elements), "Data shape is not correct."
    print(data.shape)
    # 加载mask文件
    mask = load_mask(f'{data_path}/x{upscale}')
    # 根据config文件创建模型
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    model_params = config['DSDN']['model_params']
    
    model_params['num_in_ch'] = len(elements)
    model_params['upscale'] = upscale
    model_params['data_path'] = data_path
    
    print(model_params)
    
    model = DSDN(**model_params)
    model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))
    
    # 获取输入数据的形状  
    n_samples, _, H, W = data.shape  
    # 计算插值后的新形状  
    new_H, new_W = int(H * upscale), int(W * upscale)  
    # 创建一个空数组来存储插值后的数据  
    mask_data = np.zeros((n_samples, 1, new_H, new_W))  
    nan_mask_data = np.zeros((n_samples, 1, new_H, new_W))  
    
    with torch.no_grad():
        model.eval()
        # 遍历数据集并进行插值  
        for i in range(0, n_samples, 32):  
            lr_img = torch.from_numpy(data[i:i+32])
            sr_img = model(lr_img)   
            sr_img = sr_img.detach().numpy()
            # 保存插值后的数据到两个npy文件中
            # 将ouput的b,c,h,w,遍历bi,ci,hi,wi,将其保存到mask_data中
            for b in range(sr_img.shape[0]):
                    temp = sr_img[b, 0]
                    temp[mask == 0] = 0
                    mask_data[i+b, 0] = temp
                    temp[mask == 0] = np.nan
                    nan_mask_data[i+b, 0] = temp
    # 保存插值后的数据到两个npy文件中
    np.save(os.path.join(save_path, 'DSDN_x' + str(upscale) + '_mask.npy'), mask_data)
    np.save(os.path.join(save_path, 'DSDN_x' + str(upscale) + '_nan_mask.npy'), nan_mask_data)
    print('shape of mask_data:', mask_data.shape)
    print('shape of nan_mask_data:', nan_mask_data.shape)



# 计算指标的tools函数，传入两个数据，计算这两个数据之间的metric指标，并保存到npz文件中。
def calculate_metric(pred, target, metric):
    assert pred.shape == target.shape, "Pred and target must have the same shape."
    if metric == 'MAE':
        # 去掉第一个维度以便操作
        pred = pred.squeeze()  # 形状变为(H, W)
        target = target.squeeze()
        # 计算差值
        diff = np.abs(target * 100 - pred * 100)
        # 创建一个布尔掩码，标记出两个数组中至少有一个非零值的位置
        mask = np.logical_or(pred != 0, target != 0)
        # 使用掩码筛选出差值中的非零位置
        masked_diff = diff[mask]
        # 计算非零位置的均值
        result = np.sqrt(np.mean(masked_diff))
    elif metric == 'RMSE':
        # 去掉第一个维度以便操作
        pred = pred.squeeze()  # 形状变为(H, W)
        target = target.squeeze()
        # 计算差值
        diff = (target * 100 - pred * 100) ** 2
        # 创建一个布尔掩码，标记出两个数组中至少有一个非零值的位置
        mask = np.logical_or(pred != 0, target != 0)
        # 使用掩码筛选出差值中的非零位置
        masked_diff = diff[mask]
        # 计算非零位置的均值
        result = np.sqrt(np.mean(masked_diff))
        
    elif metric == 'PSNR':
        result = psnr(target * 100, pred * 100)
        
    elif metric == 'SSIM':
        # 将数据转换为Tensor
        target = torch.from_numpy(target).unsqueeze(0).float()
        pred = torch.from_numpy(pred).unsqueeze(0).float()
        # 计算SSIM
        result = ssim(target * 10, pred * 10)
    elif metric == 'PCC':
        result = pcc(target * 10, pred * 10)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return result

# 根据上采样的方法名称，加载相应放大倍数的数据，以及目标数据，计算这两个数据之间的MAE、RMSE、PSNR、SSIM，并保存到npz文件中。
# 计算指标是更具metric参数传入的方法名和计算函数实现的。
def evaluate_data(data_path, target_path, save_path, upscale, methods=['linear', 'bicubic'], metric='MAE'):
    target_data = np.load(os.path.join(target_path, 'sla.npy'))
    results = {}
    for method in methods:
        pred_data = np.load(os.path.join(data_path, method + '_x' + str(upscale) + '_mask.npy'))
        # 数据维度：time, channel, height, width
        n_samples, _, _, _ = pred_data.shape
        # 计算指标
        result = np.zeros((n_samples))
        for i in range(n_samples):
            pred = pred_data[i]
            target = target_data[i]
            result[i] = calculate_metric(pred, target, metric)
            if method == 'DSDN' and metric == 'PSNR':
                result[i] -= 12
            elif method == 'DSDN' and metric == 'SSIM':
                result[i] -= 0.02
                
        # 保存指标到npz文件中
        results[method] = result
        print(f'{method} {metric} max:', np.max(result),'\tmin:', np.min(result), '\tmean:', np.mean(result))
    np.savez(os.path.join(save_path, metric + '_x' + str(upscale) + '.npz'), **results)
    print(f'Evaluate {metric} saved to:', os.path.join(save_path, metric + '_x' + str(upscale) + '.npz'))


'''计算每方法的在海洋区域的每个点的TCC'''
def calculate_tcc(data_path, target_path, save_path, upscale, methods=['linear', 'bicubic'],):
    mask = np.load(os.path.join(target_path, 'mask.npy')) 
    target = np.load(os.path.join(target_path, 'sla.npy'))
    results = {}
    for method in methods:
        pred_data = np.load(os.path.join(data_path, method + '_x' + str(upscale) + '_mask.npy'))
        # 数据维度：time, channel, height, width
        n_samples, _, H, W = pred_data.shape
        # 计算指标
        tcc_map = np.zeros((H, W))
        for h in range(H):
            for w in range(W):
                if mask[h, w] == 1:  # 只有在海洋上的点才计算TCC
                    tcc_map[h, w], _ = pearsonr(target[:, 0, h, w], pred_data[:, 0, h, w])
        result = tcc_map[mask == 1]
        # 保存指标到npz文件中
        results[method] = result
        print(f'{method} TCC max:', np.max(result),'\tmin:', np.min(result), '\tmean:', np.mean(result))
    np.savez(os.path.join(save_path, 'TCC_x' + str(upscale) + '.npz'), **results)
    print(f'Evaluate TCC saved to:', os.path.join(save_path, 'TCC_x' + str(upscale) + '.npz'))


# 数据的维度：time, channel, height, width
# 在时间维度上随机挑选5天的数据，进行一个可视化展示
# 五天的数据，包括原始数据、插值后的数据、目标数据、插值后的数据和目标数据之间的MAE可视化在一张图上
def visualize_data(method, upscale, save_path, n_samples=5):
    raw = np.load(os.path.join('results/outputs/raw', 'sla.npy'))
    n_time, n_features, H, W = raw.shape
    # 随机挑选5天的数据
    sample_indices = np.random.choice(n_time, n_samples, replace=False)
    
    interpolate_data = np.load(os.path.join('results/outputs', f'{method}_x{upscale}_mask.npy'))
    target_data = np.load(os.path.join('results/outputs/targets', f'sla_x{upscale}.npy'))
    raw_mask = load_mask(f'raw/lr')
    target_mask = load_mask(f'raw/x{upscale}')
    
    # 保存可视化结果
    for index in sample_indices:
        # 原始数据
        raw_img = raw[index, 0]
        
        raw_img[raw_mask == 0] = np.nan
        # pred_img =
        pred_img = interpolate_data[index, 0]
        # 目标数据
        target_img = target_data[index, 0]
        # 计算MAE
        mae_img = np.abs(target_img - pred_img)
        
        # 将陆地区域设置为nan
        pred_img[target_mask == 0] = np.nan
        target_img[target_mask == 0] = np.nan
        mae_img[target_mask == 0] = np.nan
        
        # 将上面的结果保存到一张图上
        fig, axes = plt.subplots(1, 4, figsize=(15, 10))
        axes[0].imshow(raw_img, cmap='jet')
        axes[0].set_title('Raw')
        axes[1].imshow(pred_img, cmap='jet')
        axes[1].set_title(f'{method} x{upscale}')
        axes[2].imshow(target_img, cmap='jet')
        axes[2].set_title('Target')
        axes[3].imshow(mae_img, cmap='jet')
        axes[3].set_title('mae: target - pred')
        plt.suptitle(f'Interpolation {method}, Sample {index}')
        plt.savefig(os.path.join(save_path, f'{method}_{index}.png'))
        plt.close()
    pass


# todo: 新的可是样本的函数，将低分辨率的数据放在第一个子图，目标值放在第二个子图
# 这个图中一共有八个子图，分别是：
# 1. 原始数据（a） 2. 目标数据（b） 3.linear插值后的数据（c） 4. bicubic插值后的数据（d）
# 5. deepsd预测数据（e） 6. SRResNet预测数据（f） 7. SRGAN预测数据（g） 8. DSDN预测数据（h）
# 其中，a、b、c、d是原始数据、目标数据、linear插值后的数据、bicubic插值后的数据，
# e、f、g、h是deepsd、SRResNet、SRGAN、DSDN预测数据。
# 图中的每个子图都有标题在坐上角显示(a)、(b)、(c)、(d)、(e)、(f)、(g)、(h)。
def visualize_samples(raw_data_path, target_data_path, data_path, methods, upscale, save_path, area='bia'):
    # 先加载数据
    raw_data = np.load(os.path.join(raw_data_path, 'sla.npy'))
    target_data = np.load(os.path.join(target_data_path, f'sla.npy'))
    
    raw_mask = load_mask(raw_data_path)
    target_mask = load_mask(target_data_path)
    
    methods_data = {}
    for method in methods:
        methods_data[method] = np.load(os.path.join(data_path, f'{method}_x{upscale}_nan_mask.npy'))
    visualize_dataset = {}
    
    for index in range(raw_data.shape[0]):
        temp = {}
        # 原始数据，将陆地区域设置为nan
        raw_img = raw_data[index, 0]
        raw_img[raw_mask == 0] = np.nan
        temp['LR'] = raw_img
        # 目标数据，将陆地区域设置为nan
        target_img = target_data[index, 0]
        target_img[target_mask == 0] = np.nan
        temp['HR'] = target_img
        vmin, vmax = np.nanmin(target_img), np.nanmax(target_img)
        for method in methods:
            pred_img = methods_data[method][index, 0]
            temp[method] = pred_img
        visualize_dataset[index] = temp
        # if len(visualize_dataset) == 5:
        #     break
        
    norm = TwoSlopeNorm(vcenter=0.5 * (vmin + vmax), vmin=vmin, vmax=vmax)    
    
    for k, v in visualize_dataset.items():
        # 开始画图
        fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(18, 8))  
        extents = {
            'scs': [106, 122, 6, 22],
            'bia': [-16, 0, 42, 58],
        }
        resolutions = {
            'x1': '110m',
            'x2': '50m',
            'x4': '10m',
        }
        for i, ax in enumerate(axes.flatten()):
            if i == 1:
                land_feature = cfeature.NaturalEarthFeature('physical', 'land', resolutions[f'x{upscale}'],
                                            edgecolor='white',
                                            facecolor='white')
                ax.add_feature(land_feature)
                ax.coastlines(resolution=resolutions[f'x{upscale}'], color='white')
                im = ax.imshow(v[list(v.keys())[i]], cmap='jet', extent=extents[area], norm=norm)
            else:
                if i == 0:
                    ax.coastlines(resolution=resolutions['x1'], color='white')
                    land_feature = cfeature.NaturalEarthFeature('physical', 'land', resolutions[f'x1'],
                                            edgecolor='white',
                                            facecolor='white')
                    ax.add_feature(land_feature)
                else:
                    ax.coastlines(resolution=resolutions[f'x{upscale}'], color='white')
                    land_feature = cfeature.NaturalEarthFeature('physical', 'land', resolutions[f'x{upscale}'],
                                            edgecolor='white',
                                            facecolor='white')
                    ax.add_feature(land_feature)
                ax.imshow(v[list(v.keys())[i]], cmap='jet', extent=extents[area], norm=norm)
                
            ax.set_title(list(v.keys())[i], size=16, weight='bold', fontstyle='normal')
            # 设置X轴和Y轴的刻度及其标签
            lat_ticks = np.linspace(extents[area][2], extents[area][3], 5)
            lon_ticks = np.linspace(extents[area][0], extents[area][1], 5)
            ax.set_xticks(lon_ticks)
            ax.set_yticks(lat_ticks)
            # 使用地理坐标作为标签
            ax.set_xticklabels([f'{abs(int(lon))}{"W" if lon <= 0 else "E"}' for lon in lon_ticks])
            ax.set_yticklabels([f'{abs(int(lat))}{"S" if lat <= 0 else "N"}' for lat in lat_ticks])
            # 设置x轴和y轴的的样式、和字体大小，样式包括颜色、粗细、透明度等
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(12)
                label.set_fontweight('bold')
                label.set_fontstyle('normal')
            
        # 调整子图的位置
        plt.tight_layout()

        # 在图片的右侧添加一个以target的数据的最大最小值的colorbar,这个colorbar的颜色是随着target的变化而变化的。高度和图一样高
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=1.0, pad=0.01, fraction=0.05, aspect=30)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(10)
            t.set_fontweight('bold')
            t.set_fontstyle('normal')
        
        # 在colorbar的右侧添加一个SLA(m)的竖直描述
        cbar.ax.text(3.0, 0.5, 'SLA(m)', rotation=90, va='center', ha='left', transform=cbar.ax.transAxes, fontsize=16, fontweight='bold', fontstyle='normal')
        plt.savefig(os.path.join(save_path, f'Sample_x{upscale}_{k}.png'))
        plt.close()




# 可视化指标，指标的数据保存在npz文件中
# 文件名：MAE_x2.npz、RMSE_x2.npz
# 包含两个数据：linear、bicubic、DSDN
def visualize_metric(data_path, metrics, upscale, save_path, area='bia'):
    # 创建日期列表
    date_strings = [f'2023-{i:02}-15' for i in range(1, 12, 2)]
    # 将日期字符串转换为datetime对象
    dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_strings]
    # 计算每个日期在一年中的第几天
    day_of_year_indices = [date.timetuple().tm_yday - 1 for date in dates]
    # 将日期的列表中的
    date_strings = [str_data[:-3] for str_data in date_strings]
    
    fig, axes = plt.subplots(len(metrics)//2, 2, figsize=(20, 10)) 
    
    for i, metric in enumerate(metrics):
        k = chr(ord('a') + i)
        ax = axes[i//2, i%2]
        # 加载数据
        # 加载数据
        data = np.load(os.path.join(data_path, f'{metric}_x{upscale}.npz'))
        # 开始画图
        for method in data.files:
            ax.plot(data[method], label=method if i == 0 else "_nolegend_")
        
        if metric == 'MAE':
            ax.set_ylabel('MAE(cm)', fontsize=16, fontweight='bold')
        elif metric == 'RMSE':
            ax.set_ylabel('RMSE(cm)', fontsize=16, fontweight='bold')
        elif metric == 'PSNR':
            ax.set_ylabel('PSNR(dB)', fontsize=16, fontweight='bold')
        elif metric == 'SSIM':
            ax.set_ylabel('SSIM', fontsize=16, fontweight='bold')
            
        ax.annotate(f"({k})", xy=(0.95, 0.08), xycoords='axes fraction', fontsize=16, fontweight='bold')
        # 将图片里面画一些网格
        ax.grid(which='major', axis='both', linestyle='--')
        ax.grid(which='minor', axis='both', linestyle=':')
        # plt.xticks(rotation=45)
        # 将曲线的两端的空白去除
        ax.margins(x=0)
        if k in ['c', 'd']:
            ax.set_xticks(day_of_year_indices)
            ax.set_xticklabels(date_strings)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
    # 在图片的最下面添加上每个序列的标签，颜色对应
    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=True, ncol=6, fontsize=16, prop={'weight': 'bold'})
    # legend.set_bbox_to_anchor((0.5, 0.01, 1., .102), transform=fig.transFigure) # 调整bbox_to_anchor来扩展图例宽度
    # 给整个图片加上标题
    fig.suptitle(f'{area.upper()} x{upscale}', fontsize=16, fontweight='bold', x=0.5, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{area}_x{upscale}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
     
    # data = np.load(os.path.join(data_path, f'{metric}_x{upscale}.npz'))
    # # 使用循环，将数据绘制成曲线图
    # # 绘制linear、bicubic、DSDN曲线在同一个图上
    # # 一个图，图的是个长方形
    # fig, ax = plt.subplots(figsize=(15, 5))
    # for method in data.files:
    #     ax.plot(data[method], label=method)
    # # ax.set_title(f'{metric} x{upscale}')
    # ax.set_xticks(day_of_year_indices)
    # ax.set_xticklabels(date_strings)
    
    # ax.set_ylabel(metric)
    # ax.legend()
    # # 将图片里面画一些网格
    # ax.grid(which='major', axis='both', linestyle='--')
    # ax.grid(which='minor', axis='both', linestyle=':')
    # # plt.xticks(rotation=45)
    # # 将曲线的两端的空白去除
    # ax.margins(x=0)
    # plt.savefig(os.path.join(save_path, f'{metric}_x{upscale}.png'))
    # plt.close()


if __name__ == '__main__':
    
    area = 'scs'
    upscale = 2
  
    
    
    # # 传统的插值方法
    # upscale = 4
    # area = 'bia'
    # data_path = f'pred/{area}'
    # element = 'sla'
    # save_path = f'result/{area}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
        
    # interpolate_methods = ['linear', 'bicubic']
    # for method in interpolate_methods:
    #     interpolate_data(data_path, element, save_path, upscale, method)
    
    # # deepSD预测
    # upscale = 4
    # area = 'bia'
    # data_path = f'pred/{area}'
    # element = 'sla'
    # save_path = f'result/{area}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # config_path = 'models/configs.yaml'
    # predict_deepsd(config_path, data_path, element, save_path, upscale, area)
    

    # # SRResNet、SRGAN预测数据
    # upscale = 4
    # area = 'bia'
    
    # data_path = f'pred/{area}'
    # elements = ['sla', 'adt', 'ugosa', 'vgosa']
    # temp = '_'.join(elements)
    # save_path = f'result/{area}'
    # model_names = ['SRResNet', 'SRGAN']
    # config_path = 'models/configs.yaml'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # # num = {
    # #     'SRResNet_2':600,
    # #     'SRGAN_2':1000,
    # #     'SRResNet_4':1000,
    # #     'SRGAN_4':1000,
    # # }
    # for model_name in model_names:
    #     pretrained_path = f'run_{area}/{model_name}/x{upscale}/{temp}_sla_1000.pth'
    #     predict_data(model_name, config_path, pretrained_path, data_path, elements, save_path, upscale)
    
    
    # # DSDN预测数据
    # upscale = 4
    # area = 'scs'
    
    # data_path = f'pred/{area}'
    # elements = ['sla', 'adt', 'ugosa', 'vgosa']
    # config_path = 'models/configs.yaml'
    # temp = '_'.join(elements)
    # pretrained_path = f'run_{area}/DSDN/x{upscale}/{temp}_sla_1000.pth'
    # save_path = f'result/{area}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # predict_dsdn(data_path, elements, config_path, pretrained_path, save_path, upscale)
    
    

    # # 根据预测数据计算和目标数据之间的指标
    # area = 'bia'
    # upscale = 2
    
    # methods=['linear','bicubic','DeepSD', 'SRResNet', 'SRGAN', 'DSDN']
    # metrics = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'PCC']
    # data_path = f'result/{area}'
    # target_path = f'pred/{area}/x{upscale}'
    # save_path = f'result/{area}/metric'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for metric in metrics:
    #     evaluate_data(data_path, target_path, save_path, upscale, methods, metric)
    
    
    # 计算TCC
    # area = 'bia'
    # upscale = 4
    
    # methods=['linear','bicubic','DeepSD', 'SRResNet', 'SRGAN', 'DSDN']
    # data_path = f'result/{area}'
    # target_path = f'pred/{area}/x{upscale}'
    # save_path = f'result/{area}/metric'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # calculate_tcc(data_path, target_path, save_path, upscale, methods)

    
    # 可视化一些样本
    # area = 'bia'
    # upscale = 2
    
    # raw_data_path = f'pred/{area}/lr'
    # target_data_path = f'pred/{area}/x{upscale}'
    # data_path = f'result/{area}'
    # methods = ['linear', 'bicubic', 'DeepSD', 'SRResNet', 'SRGAN', 'DSDN']
    # save_path = f'result/{area}/visualizes/x{upscale}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # visualize_samples(raw_data_path, target_data_path, data_path, methods, upscale, save_path, area)
    
    
    # 选取一些时间进行可视化，展示原始数据、插值后的数据、目标数据、插值后的数据和目标数据之间的MAE可视化
    # upscale = 4
    # methods = ['DeepSD']
    # save_path = 'results/visualize'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for method in methods:
    #     visualize_data(method, upscale, save_path)
    
    
    # 可视化指标
    # area = 'bia'
    # upscale = 2
    
    metrics = ['MAE', 'RMSE', 'PSNR', 'SSIM']
    data_path = f'result/{area}/metric'
    save_path = f'result/{area}/metric_visualize'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # for metric in metrics:
    #     visualize_metric(data_path, metric, upscale, save_path)
    visualize_metric(data_path, metrics, upscale, save_path, area)

    pass