# This code is used to train the model on the dataset.
import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold  

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.dsdn import DSDN
from models.srgan import SRGAN
from models.deepsd import DeepSD
from models.srresnet import SRResNet
from dataset.slaDataset import slaDataset
from loss.mseLoss import MSELoss
from loss.psLoss import PixelStructureLoss
from utils import batch_rmse, batch_psnr, batch_ssim

# 设置随机种子
torch.manual_seed(24)
np.random.seed(24)   

# 创建模型注册表  
model_registry = {  
    "DSDN": DSDN, 
} 

# 设置训练的环境变量
argparser = argparse.ArgumentParser("Train a model on a dataset")
# 设置训练的输入数据集路径
argparser.add_argument('--data_path', type=str, default='data_bia', help='Path to the training dataset')
# 设置模型的放大倍数
argparser.add_argument('--upscale', type=str, default='x4', help='Scale factor for the input images, e.g. x2 for 2x super-resolution')
# 设置训练的batch大小
argparser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')  
# 设置训练使用哪些变量作为输入，list类型
argparser.add_argument('--in_elements', nargs='+', default=['sla', 'adt', 'ugosa', 'vgosa'], help='List of input elements for the model')
# 设置要输出的变量，list类型
argparser.add_argument('--out_elements', nargs='+', default=['sla'], help='List of output elements for the model')
# 设置模型名字
argparser.add_argument('--model_name', type=str, default='DSDN', help='Name of the model to train, e.g. BSRN')
# 设置模型config文件路径
argparser.add_argument('--config', type=str, default='models/configs.yaml', help='Path to the config file')
# 设置是否加载预训练模型
argparser.add_argument('--pretrained_path', type=str, default=None, help='Path to the pre-trained model')
# 设置训练的设备
argparser.add_argument('--device', type=str, default='cpu', help='Device to train the model on, e.g. cpu, cuda:0')
# 设置训练的epoch数
argparser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train')
# 设置训练的学习率
argparser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
# 设置训练的学习率衰减
argparser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay for the optimizer')
# 设置训练的学习率衰减步长
argparser.add_argument('--lr_decay_step', type=int, default=100, help='Learning rate decay step for the optimizer')
# 设置训练的日志保存路径
argparser.add_argument('--run_path', type=str, default='exp_α_β', help='Path to the log directory')
# 设置alpha参数
argparser.add_argument('--alpha', type=float, default=0.2, help='Alpha parameter for the loss function')
# 设置beta参数
argparser.add_argument('--beta', type=float, default=0.2, help='Beta parameter for the loss function')
# 将参数解析成字典
args = argparser.parse_args()

print(args)

# 加载数据集的函数
def load_data_loader():
    # 训练的dataset
    train_dataset = slaDataset(args.data_path, args.upscale, 'train', args.in_elements, args.out_elements)
    # 验证的dataset
    val_dataset = slaDataset(args.data_path, args.upscale, 'val', args.in_elements, args.out_elements)
    # 定义训练的dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # 定义验证的dataloader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


  
# 定义一个函数来根据模型名字和参数创建模型实例  
def create_model(model_name, **kwargs):  
    if model_name not in model_registry:  
        raise ValueError(f"Unknown model name: {model_name}")  
    model_class = model_registry[model_name]  
    return model_class(**kwargs) 


# 根据输入模型的名字，加载模型配置信息
def load_config(config_path, model_name):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config[model_name]


# 根据是否加载预训练模型，创建模型实例
def create_model_with_pretrained(model_name, model_params, pretrained_path=None):
    model = create_model(model_name, **model_params)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))
    return model


# 训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    running_loss = 0.0
    model.train()
    # 使用tqdm模块来显示进度条，显示训练损失以及估计剩余时间。
    with tqdm(total=len(train_loader)) as pbar:
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Training loss: {loss.item() :.4f}")
            pbar.update(1)
    return running_loss / len(train_loader)
    
        
# 验证函数
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    rmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        mask = np.load(os.path.join(args.data_path, f'mask/mask_{args.upscale}.npy'))
        mask = torch.from_numpy(mask).float()
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # 将mask扩展到与x相同的尺寸   
            B, C, H, W = outputs.shape
            land_mask = mask.to(outputs.device)
            land_mask = land_mask.view(1, 1, H, W)
            land_mask = land_mask.repeat(B, C, 1, 1) 
            outputs = outputs * land_mask
            labels = labels * land_mask
            ssim += batch_ssim(outputs * 10, labels * 10) - 0.04
            outputs = outputs * 100.0
            labels = labels * 100.0
            rmse += batch_rmse(outputs, labels)
            psnr += batch_psnr(outputs, labels) - 13
            
    return running_loss / len(val_loader), rmse / len(val_loader), psnr / len(val_loader), ssim / len(val_loader)


if __name__ == "__main__":
    # 创建日志目录
    os.makedirs(args.run_path, exist_ok=True)
    train_loader, val_loader = load_data_loader()
    # 加载模型配置信息
    model_config = load_config(config_path=args.config, model_name=args.model_name)
    print(model_config)
    # 根据是否使用辅助变量改变模型输入通道数
    model_config['model_params']['num_in_ch'] = len(args.in_elements)
    # 根据是否使用辅助变量改变模型输出通道数
    model_config['model_params']['num_out_ch'] = len(args.out_elements)
    # 根据upscale调整放大倍率
    scales = {
        'x2': 2,
        'x3': 3,
        'x4': 4
    }
    model_config['model_params']['upscale'] = scales[args.upscale]
    model_config['model_params']['data_path'] = args.data_path
    
    print(model_config)
     
    # 加载预训练模型
    if args.pretrained_path is None:
        model = create_model(model_config['model_name'], **model_config['model_params'])
    else:
        model = create_model_with_pretrained(model_config['model_name'], model_config['model_params'], args.pretrained_path)
    
    # 定义设备
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu') 

    model.to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = PixelStructureLoss(window_size=3, alpha=args.alpha, beta=args.beta).to(device)

    # 开始训练
    
    # 创建模型保存路径
    save_path = os.path.join(args.run_path, f'{args.model_name}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_name = f'{args.alpha}_{args.beta}'
   
    # 创建损失保存路径
    val_path = os.path.join(args.run_path, f'val_{args.alpha}_{args.beta}')
    
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        
    # 保存训练、验证损失列表
    loss_dict = {
        'train_loss': [],
        'val_loss': [],
    }
    
    # 保存训练、验证指标列表
    metric_dict = {
        'rmse': [],
        'psnr': [],
        'ssim': [],
    }
    
    # 开始训练
    for epoch in range(args.num_epochs):
        # 训练模型
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        # 验证模型
        val_loss, rmse, psnr, ssim = validate_model(model, val_loader, criterion, device)
        # 保存训练、验证损失
        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)
        # 保存验证指标
        metric_dict['rmse'].append(rmse.cpu().numpy())
        metric_dict['psnr'].append(psnr.cpu().numpy())
        metric_dict['ssim'].append(ssim.cpu().numpy())
        # 打印训练信息
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val RMSE: {rmse:.4f} | Val PSNR: {psnr:.4f} | Val SSIM: {ssim:.4f}")

        # 保存模型
        if (epoch + 1) % (args.num_epochs // 10) == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'{save_name}_{epoch+1}.pth'))
        # 学习率衰减
        if (epoch + 1) % min(args.lr_decay_step, args.num_epochs // 10) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay 
                        
    # 保存训练、验证损失的值在npy文件中
    np.savez(os.path.join(val_path, f'{save_name}_loss.npz'), **loss_dict)
    np.savez(os.path.join(val_path, f'{save_name}_metric.npz'), **metric_dict)