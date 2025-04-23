'''绘制研究区域地图'''
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 

# fig = plt.figure(figsize=(15, 10))
# proj = ccrs.PlateCarree()


# ax = fig.add_axes([0.3, 0, 0.5, 0.7], projection=proj)

# # 设置经纬度范围,限定为中国
# # 注意指定crs关键字,否则范围不一定完全准确
# extents = [-20, 130, 0, 60]
# ax.set_extent(extents)
# # 添加各种特征
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.LAND, edgecolor='gray')
# ax.add_feature(cfeature.LAKES, edgecolor='gray')
# ax.add_feature(cfeature.RIVERS)
# ax.add_feature(cfeature.BORDERS,edgecolor='gray')

# ax.text(-16, 48, 'Biscay-Ireland\n      Atlantic', transform=ccrs.PlateCarree(), fontsize=8, color='blue')
# ax.text(108, 11, 'South China\n       Sea', transform=ccrs.PlateCarree(), fontsize=8, color='blue')

# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--',linewidth=0)
# gl.xlabels_top = False  # 不显示顶部的经度标签
# gl.ylabels_left = False  # 不显示右侧的纬度标签
# gl.ylabels_right = False  # 不显示右侧的纬度标签
# gl.xlabel_style = {'size': 10}  # 设置横坐标标签的大小
# gl.ylabel_style = {'size': 10}  # 设置纵坐标标签的大小


# ax2=fig.add_axes([0,0.2,0.3,0.3], projection=proj)
# extents_2 = [-16, 0, 42, 58]
# ax2.set_extent(extents_2, crs=proj)
# ax2.add_feature(cfeature.OCEAN)
# ax2.add_feature(cfeature.LAND, edgecolor='gray')
# ax2.add_feature(cfeature.LAKES, edgecolor='gray')
# ax2.add_feature(cfeature.RIVERS)
# ax2.add_feature(cfeature.BORDERS,edgecolor='gray')

# ax2.text(-12, 49, 'Biscay-Ireland\n      Atlantic', transform=ccrs.PlateCarree(), fontsize=14, color='blue')
# ax2.text(0.95, 0.05, '(a)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')
# gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--',linewidth=0)
# gl.xlabels_top = False  # 不显示顶部的经度标签
# gl.ylabels_right = False  # 不显示右侧的纬度标签
# gl.xlabel_style = {'size': 10}  # 设置横坐标标签的大小
# gl.ylabel_style = {'size': 10}  # 设置纵坐标标签的大小



# ax1=fig.add_axes([0.80,0.2,0.3,0.3], projection=proj)
# extents_1 = [106, 122, 6, 22]
# ax1.set_extent(extents_1, crs=proj)
# ax1.add_feature(cfeature.OCEAN)
# ax1.add_feature(cfeature.LAND, edgecolor='gray')
# ax1.add_feature(cfeature.LAKES, edgecolor='gray')
# ax1.add_feature(cfeature.RIVERS)
# ax1.add_feature(cfeature.BORDERS,edgecolor='gray')

# ax1.text(111, 13, 'South China\n       Sea', transform=ccrs.PlateCarree(), fontsize=14, color='blue')
# ax1.text(0.95, 0.05, '(b)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')
# gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--',linewidth=0)
# gl.xlabels_bottom = False  # 不显示顶部的经度标签
# gl.ylabels_left = False  # 不显示右侧的纬度标签
# gl.xlabel_style = {'size': 10}  # 设置横坐标标签的大小
# gl.ylabel_style = {'size': 10}  # 设置纵坐标标签的大小

# mark_inset(ax,ax1,loc1=1,loc2=2,alpha=1,fc='none',ec='k',ls=(0, (5, 5)),lw=2)
# mark_inset(ax,ax1,loc1=3,loc2=4,alpha=1,fc='none',ec='k',ls=(0, (5, 5)),lw=2)

# mark_inset(ax,ax2,loc1=1,loc2=2,alpha=1,fc='none',ec='k',ls=(0, (5, 5)),lw=2)
# mark_inset(ax,ax2,loc1=3,loc2=4,alpha=1,fc='none',ec='k',ls=(0, (5, 5)),lw=2)

# # 保存图片
# plt.savefig('temp.pdf', dpi=300, bbox_inches='tight', format='pdf')



'''绘制不同指标的箱图分析结果'''

# from matplotlib.gridspec import GridSpec
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd


# # 使用np加载npz文件
# area = 'scs'
# upscale = 4
# data_path = f'result/{area}/metric'
# metric = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'TCC']

# data = {}

# # 加载npz文件，打印每个方法的结果

# for m in metric:
#     data[m] = np.load(f'{data_path}/{m}_x{upscale}.npz', allow_pickle=True)

# # # Create subplots
# # fig, axs = plt.subplots(3, 2, figsize=(15, 10))


# # Plot MAE
# mae_data = {}
# for k, v in data['MAE'].items():
#     mae_data[k] = v
# mae_data = pd.DataFrame(mae_data)

# # # axs[0, 0].boxplot(mae_data, showfliers=False)
# # # axs[0, 0].set_ylabel('MAE')
# # # axs[0, 0].text(0.95, 0.05, '(a)', transform=axs[0, 0].transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')


# # Plot RMSE
# rmse_data = {}
# for k, v in data['RMSE'].items():
#     rmse_data[k] = v
# rmse_data = pd.DataFrame(rmse_data)

# # # axs[0, 1].boxplot(rmse_data, showfliers=False)
# # # axs[0, 1].set_ylabel('RMSE')
# # # axs[0, 1].text(0.95, 0.05, '(b)', transform=axs[0, 1].transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')


# # Plot PSNR
# psnr_data = {}
# for k, v in data['PSNR'].items():
#     psnr_data[k] = v
# psnr_data = pd.DataFrame(psnr_data)

# # # axs[1, 0].boxplot(psnr_data, showfliers=False)
# # # axs[1, 0].set_ylabel('PSNR')
# # # axs[1, 0].text(0.95, 0.05, '(c)', transform=axs[1, 0].transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

# # Plot SSIM
# ssim_data = {}
# for k, v in data['SSIM'].items():
#     ssim_data[k] = v
# ssim_data = pd.DataFrame(ssim_data)

# # # axs[1, 1].boxplot(ssim_data, showfliers=False)
# # # axs[1, 1].set_ylabel('SSIM')
# # # axs[1, 1].text(0.95, 0.05, '(d)', transform=axs[1, 1].transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

# # Plot TCC
# tcc_data = {}
# for k, v in data['TCC'].items():
#     tcc_data[k] = v
# tcc_data = pd.DataFrame(tcc_data)

# # # axs[2, 0].boxplot(pcc_data, showfliers=False)
# # # axs[2, 0].set_ylabel('PCC')
# # # axs[2, 0].text(0.95, 0.05, '(e)', transform=axs[2, 0].transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

# # # Hide empty subplot
# # # axs[2, 1].axis('off')

# # # # Add labels
# # # for ax in axs.flat:
# # #     ax.set(xlabel='Models')


# # 标签
# labels = list(mae_data.keys())
# # 定义每种方法的颜色
# colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFFF99']

# # 创建一个 figure 对象
# fig = plt.figure(figsize=(8, 11))
# # 使用 GridSpec 控制子图布局
# gs = GridSpec(3, 2, figure=fig)  # 三行两列
# # 第一行：MAE 子图
# ax1 = fig.add_subplot(gs[0, 0])
# bp1 = ax1.boxplot(mae_data, patch_artist=True, showfliers=False)
# for patch, color in zip(bp1['boxes'], colors):
#     patch.set_facecolor(color)  # 设置每个方法的颜色
# ax1.set_ylabel('MAE', fontsize=12)
# ax1.tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
# # 第二行：RMSE 子图
# ax2 = fig.add_subplot(gs[0, 1])
# bp2 = ax2.boxplot(rmse_data, patch_artist=True, showfliers=False)
# for patch, color in zip(bp2['boxes'], colors):
#     patch.set_facecolor(color)  # 设置每个方法的颜色
# ax2.set_ylabel('RMSE', fontsize=12)
# ax2.tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
# ax3 = fig.add_subplot(gs[1, 0])
# bp3 = ax3.boxplot(psnr_data, patch_artist=True, showfliers=False)
# for patch, color in zip(bp3['boxes'], colors):
#     patch.set_facecolor(color)  # 设置每个方法的颜色
# ax3.set_ylabel('PSNR', fontsize=12)
# ax3.tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
# ax4 = fig.add_subplot(gs[1, 1])
# bp4 = ax4.boxplot(ssim_data, patch_artist=True, showfliers=False)
# for patch, color in zip(bp4['boxes'], colors):
#     patch.set_facecolor(color)  # 设置每个方法的颜色
# ax4.set_ylabel('SSIM', fontsize=12)
# ax4.tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
# # 第三行：PSNR 子图（占满整行）
# ax5 = fig.add_subplot(gs[2, :])  # 使用 gs[2, :] 让子图占满整行
# bp5 = ax5.boxplot(tcc_data, labels=labels, patch_artist=True, showfliers=False)
# for patch, color in zip(bp5['boxes'], colors):
#     patch.set_facecolor(color)  # 设置每个方法的颜色
# ax5.set_ylabel('TCC', fontsize=12)
# plt.tight_layout()
# plt.show()






# 加载npy文件，打印其中的min、max、mean、std值
# import torch
# import numpy as np
# from scipy.ndimage import zoom  
# import matplotlib.pyplot as plt
# from models.dsdn import DSDN
# from train import create_model, create_model_with_pretrained, load_config, model_registry, args

# data = np.load('data/lr/sla_20230101.npy')

# print('min:', np.min(data))
# print('max:', np.max(data))
# print('mean:', np.mean(data))
# print('std:', np.std(data))
# print('shape:', data.shape)
# print('dtype:', data.dtype)
# print('type:', type(data))

# # 将npy数据进行可视化
# mask = np.load('raw/x2/mask.npy')

# data[mask==0] = np.nan
# plt.imshow(data, cmap='jet')
# plt.clim(-0.3, 0.3)
# plt.colorbar()
# plt.show()

# # 加载mask文件，打印其中的min、max、mean、std值
# mask = np.load('raw/x2/mask.npy')

# print('min:', np.min(mask))
# print('max:', np.max(mask))
# print('mean:', np.mean(mask))
# print('std:', np.std(mask))
# print('shape:', mask.shape)
# print('dtype:', mask.dtype)
# print('type:', type(mask))
# print('sum:', np.sum(mask))

# # 将mask数据进行可视化
# # plt.imshow(mask, cmap='gray')
# # plt.show()    
# # data = data
# data = zoom(data, (2, 2), mode='reflect', order=1)
# print(data.shape)
# data[mask==0] = np.nan

# print(np.nanmin(data))
# print(np.nanmax(data))
# print(np.nanmean(data))
# print(np.nanstd(data))

# plt.imshow(data, cmap='jet')
# plt.clim(-0.3, 0.3)
# plt.colorbar()
# plt.show()


# # 读取原先数据的二倍放大的真实值
# true_data = np.load('data/x2/sla_20230101.npy')[0]

# data_interp = zoom(data[0], (2, 2), mode='reflect', order=3)

# # 将数据中的陆地区域的设置为0
# true_data[mask==0] = 0
# data_interp[mask==0] = 0

# # 将数据的单位从m转为cm
# data_interp = data_interp
# true_data = true_data
# # 计算rmse
# rmse = np.sqrt(np.mean((true_data - data_interp)**2))
# print('rmse:', rmse)

# 可视化数据，将可是话三个子图，分别显示插值、真实值、误差MAE
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(data_interp, cmap='jet')
# axes[0].set_title('Interpolated')
# axes[1].imshow(true_data, cmap='jet')
# axes[1].set_title('True')
# axes[2].imshow(np.abs(true_data - data_interp), cmap='jet')
# axes[2].set_title('MAE')
# plt.show()

# model_config = load_config(config_path=args.config, model_name=args.model_name)
# print(model_config)
# args.pretrained_path = 'run/DSDN/sla_sla_1000.pth'
# model = create_model_with_pretrained(model_config['model_name'], model_config['model_params'], pretrained_path=args.pretrained_path)
# model.to(args.device)

# # 将数据转为tensor、将数据扩充一个Batch维度
# data = torch.from_numpy(data).unsqueeze(0).to(args.device)
# print(data.shape)

# # 使用将数据输入模型，得到输出
# with torch.no_grad():
#     output = model(data)

# # 将输出转为numpy
# output = output.squeeze(0).cpu().numpy()

# output = output[0] * 100
# output[mask==0] = 0
# # 计算rmse
# rmse = np.sqrt(np.mean((true_data - output)**2))
# print('rmse:', rmse)

# true_data = true_data / 100
# output = output / 100
# # 答应真值、output、的最值
# print('max:', np.max(true_data))
# print('min:', np.min(true_data))
# print('mean:', np.mean(true_data))
# print('std:', np.std(true_data))
# print('max:', np.max(output))
# print('min:', np.min(output))
# print('mean:', np.mean(output))
# print('std:', np.std(output))

# # 可视化数据，将可是话三个子图，分别显示插值、真实值、误差MAE
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# im = axes[0].imshow(output, cmap='jet')
# axes[0].set_title('DSDN')
# # 设置一下clim，使得颜色范围在-0.3到0.3之间
# im.set_clim(-0.3, 0.3)

# im2 = axes[1].imshow(true_data, cmap='jet')
# axes[1].set_title('True')
# im2.set_clim(-0.3, 0.3)
# im3 = axes[2].imshow(np.abs(true_data - output), cmap='jet')
# axes[2].set_title('MAE')
# im3.set_clim(0, 0.1)
# plt.show()


# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# # 创建链表节点
# node1 = ListNode(1)
# node2 = ListNode(2)
# node3 = ListNode(3)

# # 将节点连接成一个链表: 1 -> 2 -> 3
# node1.next = node2
# node2.next = node3

# # 遍历链表并打印每个节点的值
# current_node = node1
# while current_node is not None:
#     print(current_node.val)
#     # 访问下一个节点
#     current_node = current_node.next


# def isValid(s: str) -> bool:
#     if len(s) <= 1:
#         return False
#     left_ = []
#     for i in range(len(s)):
#         if s[i] in ['(','[','{']:
#             left_.append(s[i])
#             continue
#         else:
#             if len(left_) != 0:
#                 temp = left_.pop()
#                 if s[i] == ')' and temp == '(':
#                     continue
#                 elif s[i] == ']' and temp == '[':
#                     continue
#                 elif s[i] == '}' and temp == '{':
#                     continue    
#                 else:
#                     return False
#     if len(left_) == 0:
#         return True
#     return False
    
# if __name__ == '__main__':
#     s = "([])"
#     print(isValid(s))

# 将字符转ASCII
# print(ord('a') - ord('e'))
# print(ord('d') - ord('g'))
# def char_to_ascii(s: str) -> int:
#     return ord(s)


''''绘制测试集上的结果之间的平均绝对值偏差分布图'''
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 使用np加载npy数据
area = 'bia'
upscale = 4


data_path = f'result/{area}'
target_path = f'pred/{area}/x{upscale}'
save_path = f'result/{area}/mae_map'
if not os.path.exists(save_path):
    os.makedirs(save_path)
methods = ['linear', 'bicubic', 'DeepSD', 'SRResNet', 'SRGAN', 'DSDN']



# # 计算每个方法的绝对值偏差，并保存到字典中
# target_data = np.load(os.path.join(target_path, 'sla.npy'))
# # mask = np.load(os.path.join(target_path, 'mask.npy'))

# methods_mae = {}
# for method in methods:
#     temp = []
#     output_data = np.load(f'{data_path}/{method}_x{upscale}_mask.npy')
#     for i in range(output_data.shape[0]):
#         target = target_data[i, 0]
#         # target[mask==0] = 0
#         temp.append(np.abs(target - output_data[i, 0]) * 100)  
#     method_mae = np.mean(temp, axis=0)
#     if method == 'linear':
#         vmin, vmax = np.nanmin(method_mae), np.nanmax(method_mae)
#     print(f'{method} mae map min: {np.min(method_mae)}, max: {np.max(method_mae)}')
#     print(f'{method} mae map mean: {np.mean(method_mae)}, std: {np.std(method_mae)}')
#     methods_mae[method] = method_mae

# methods_mae['vmin'] = vmin
# methods_mae['vmax'] = vmax

# # 将methods_mae中的数据保存到npz文件中

# np.savez(f'{save_path}/mae_map_{area}_x{upscale}.npz', **methods_mae)



# 读取npz文件中的数据
mae_map = np.load(f'{save_path}/mae_map_{area}_x{upscale}.npz')
norm = TwoSlopeNorm(vcenter=0.5 * (mae_map['vmin'] + mae_map['vmax']), vmin=mae_map['vmin'], vmax=mae_map['vmax'])   
# 一共有6个方法，分别画图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
extents = {
    'scs': [106, 122, 6, 22],
    'bia': [-16, 0, 42, 58],
}

for i, method in enumerate(methods):
    if i == 0:
        im = axes[i//3, i%3].imshow(mae_map[method], cmap='jet', aspect='auto', extent=extents[area], norm=norm)
    else:
        axes[i//3, i%3].imshow(mae_map[method], cmap='jet', aspect='auto', extent=extents[area], norm=norm)
    axes[i//3, i%3].set_xlabel(method)
    axes[i//3, i%3].grid(True)
    # 设置X轴和Y轴的刻度及其标签
    lat_ticks = np.linspace(extents[area][2], extents[area][3], 5)
    lon_ticks = np.linspace(extents[area][0], extents[area][1], 5)
    axes[i//3, i%3].set_xticks(lon_ticks)
    axes[i//3, i%3].set_yticks(lat_ticks)
    # 使用地理坐标作为标签
    axes[i//3, i%3].set_xticklabels([f'{abs(int(lon))}{"W" if lon <= 0 else "E"}' for lon in lon_ticks])
    axes[i//3, i%3].set_yticklabels([f'{abs(int(lat))}{"S" if lat <= 0 else "N"}' for lat in lat_ticks])


plt.tight_layout()

# 添加一个colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=1.0, pad=0.01, fraction=0.05, aspect=30)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(10)
    t.set_fontweight('bold')
    t.set_fontstyle('normal')

# 在colorbar的右侧添加一个SLA(m)的竖直描述
cbar.ax.text(2.5, 0.5, 'Absolute Bias (cm)', rotation=90, va='center', ha='left', transform=cbar.ax.transAxes, fontsize=16, fontweight='bold', fontstyle='normal')

plt.savefig(f'{save_path}/mae_map_{area}_x{upscale}.eps' , format='eps', dpi=300, bbox_inches='tight')


'''计算测试集上两个数据之间海洋数据的每个点的TCC'''
# import numpy as np
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt

# # 假设data1和data2是两个维度为(T, 1, H, W)的numpy数组
# mask = np.load('pred/bia/x2/mask.npy')   
# data1 = np.load('pred/bia/x2/sla.npy')   
# data2 = np.load('result/bia/linear_x2_mask.npy')

# _, _, H, W = data1.shape

# # 计算TCC map
# tcc_map = np.zeros((H, W))
# for h in range(H):
#     for w in range(W):
#         if mask[h, w] == 1:  # 只有在海洋上的点才计算TCC
#             tcc_map[h, w], _ = pearsonr(data1[:, 0, h, w], data2[:, 0, h, w])

# # 应用mask
# masked_tcc_map = np.ma.array(tcc_map, mask=(mask == 0))

# # 绘制TCC map
# plt.figure(figsize=(8, 6))
# cax = plt.imshow(masked_tcc_map, cmap='coolwarm', vmin=-1, vmax=1)
# plt.colorbar(cax)
# plt.title('TCC Map with Land Masked')
# plt.show()

# 应用mask并提取海洋区域的TCC值
# ocean_tcc_values = tcc_map[mask == 1]
# print('shape:', ocean_tcc_values.shape)
# print('min:', np.min(ocean_tcc_values))
# print('max:', np.max(ocean_tcc_values))
# print('mean:', np.mean(ocean_tcc_values))
# print('std:', np.std(ocean_tcc_values))

# # 统计海洋区域的TCC值分布
# plt.figure(figsize=(14, 5))

# # TCC Map
# plt.subplot(1, 3, 1)
# masked_tcc_map = np.ma.array(tcc_map, mask=(mask == 0))
# cax = plt.imshow(masked_tcc_map, cmap='coolwarm', vmin=0.9, vmax=1)
# plt.colorbar(cax)
# plt.title('TCC Map with Land Masked')

# # 海洋区域TCC值的分布
# plt.subplot(1, 3, 2)
# plt.hist(ocean_tcc_values, bins=30, color='skyblue', edgecolor='black')
# plt.title('Distribution of TCC Values in Ocean Areas')
# plt.xlabel('TCC Value')
# plt.ylabel('Frequency')

# plt.subplot(1, 3, 3)
# plt.boxplot(ocean_tcc_values, showfliers=False)  # horizontal box plot
# plt.title('Box Plot of TCC Values in Ocean Areas')
# plt.xlabel('TCC Value')

# plt.tight_layout()
# plt.show()

'''绘制等值线图用于讨论，将对一个涡旋结构进行时间维度观察'''

# import matplotlib.pyplot as plt
# import numpy as np

# # 生成示例数据（二维高斯分布）
# x = np.linspace(0, 16, 128)
# y = np.linspace(0, 16, 128)
# X, Y = np.meshgrid(x, y)

# Z = np.load('result/bia/DSDN_x2_mask.npy')[0, 0]

# print('shape:', X.shape, Y.shape, Z.shape)
# # 绘制等值线图
# plt.figure(figsize=(8, 6))
# contour = plt.contour(X, Y, Z, levels=[0.2], 
#                      colors='red',    # 设置等值线颜色
#                      linewidths=2,    # 设置线宽
#                      linestyles='--') # 设置线型

# # 自动标注等值线（方法一）
# plt.clabel(contour, inline=True, fontsize=10, fmt={0.3: '0.3'})

# # 或手动精确标注（方法二，选择其一）
# # for line in contour.collections[0].get_paths():
# #     verts = line.vertices  # 获取等值线坐标
# #     # 计算标注位置（这里取几何中心）
# #     x_center = np.mean(verts[:,0])
# #     y_center = np.mean(verts[:,1])
# #     plt.text(x_center, y_center, '0.3',
# #             color='red', fontsize=12,
# #             ha='center', va='center',
# #             backgroundcolor='white')

# # 添加辅助元素
# plt.title('Contour Plot with 0.3 Level Highlighted')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.grid(True, linestyle='--', alpha=0.7)

# # plt.colorbar(label='Z Value')  # 添加颜色条（可选）

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.path import Path
# import cartopy.crs as ccrs  # 用于地图投影
# from matplotlib.colors import ListedColormap
# from matplotlib.colors import TwoSlopeNorm


# # 模拟示例数据（假设陆地值为0，海洋值非零）
# # 替换为你的实际数据加载代码
# lon = np.linspace(106, 122, 64)  # 经度范围示例
# lat = np.linspace(6, 22, 64)    # 纬度范围示例



# # 数据预处理：将陆地值（0）转换为NaN
# sla_masked = np.load('pred/scs/lr/sla.npy')[0, 0]
# # sla_masked = np.flipud(sla_masked)  # 纬度方向翻转，以符合地理坐标
# vmin, vmax = np.nanmin(sla_masked), np.nanmax(sla_masked)  # 计算最小最大值
# norm = TwoSlopeNorm(vcenter=0.5 * (vmin + vmax), vmin=vmin, vmax=vmax) 
# # 创建地图投影
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
# ax.coastlines()  # 添加海岸线
# # ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
# ax.gridlines(draw_labels=True)
# # 定义涡旋强度层级（关键改进点）
# levels = [-0.05, 0.3]  # 自定义阈值
 
# # # 创建自定义颜色映射
# # colors = ['#3182bd', '#6baed6', '#bdd7e7',  # 负值冷色调
# #           '#ffffbf',                         # 零值中性色
# #           '#fee0d2', '#fcbba1', '#fc9272']  # 正值暖色调

# # cmap = ListedColormap(colors)

# # contour = ax.contourf(lon, lat, sla_masked,
# #                      cmap='jet',
# #                      extend='both', norm=norm,)
# im = ax.imshow(sla_masked, cmap='jet', extent=[lon.min(), lon.max(), lat.min(), lat.max()], norm=norm)
# # # 绘制等值线
# # contours = ax.contour(
# #     lon, lat, sla_masked,
# #     levels=levels,
# #     colors=('red', 'blue'),  # 正负涡旋不同颜色
# #     linewidths=1.5,
# #     linestyles='--'
# # )

# cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
# cbar.set_label('Sea Level Anomaly (m)')



# sla_masked = np.flipud(sla_masked)  # 纬度方向翻转，以符合地理坐标
# # 生成等值线并筛选闭合曲线
# cs = ax.contour(lon, lat, sla_masked, levels=levels, colors='black', linewidths=0.8)
 
# # 仅保留闭合等值线并添加标签
# closed_contours = []

# for i, collection in enumerate(cs.collections):  # 添加索引获取对应层级值
#     level = levels[i]  # 直接使用预定义的数值层级
#     for path in collection.get_paths():
        
#         if len(path.codes) > 0 and path.codes[-1] == Path.CLOSEPOLY:
#             closed_contours.append(path)
#             # # 添加数值标签
#             # x_center = np.mean(path.vertices[:, 0])
#             # y_center = np.mean(path.vertices[:, 1])
#             # ax.text(x_center, y_center, 
#             #        f'{level:.2f}',
#             #        fontsize=8,
#             #        ha='center', 
#             #        va='center',
#             #        color='blue' if level < 0 else 'red',
#             #        backgroundcolor='none')
 
# # 清除原始等值线（保留筛选后的闭合线）
# for collection in cs.collections:
#     collection.remove()
 
# # 重新绘制筛选后的闭合等值线
# closed_cs = ax.contour(lon, lat, sla_masked, levels=levels, 
#                       colors='black', linewidths=1.2,
#                       linestyles='solid')


# plt.clabel(closed_cs, inline=True, fontsize=10, fmt={-0.05: '-0.05', 0.3: '0.3'})

# # 自动标注闭合等值线（涡旋）
# def find_closed_contours(contour_collection):
#     """检测并返回闭合的等值线"""
#     closed_contours = []
#     for level in contour_collection.collections:
#         for path in level.get_paths():
#             # 检查闭合的两种方法（按优先级排列）
#             if hasattr(path, 'closed') and path.closed:
#                 closed_contours.append(path.vertices)
#             elif len(path.codes) > 0 and path.codes[-1] == Path.CLOSEPOLY:
#                 closed_contours.append(path.vertices)
#     return closed_contours

# # 获取闭合等值线
# closed_contours = find_closed_contours(contours)




# # 为每个闭合等值线添加标注
# for i, contour in enumerate(closed_contours):
#     # 计算几何中心
#     x_center = np.mean(contour[:, 0])
#     y_center = np.mean(contour[:, 1])
#     # 添加标注文本
#     ax.text(
#         x_center, y_center,
#         f'V{i+1}',  # 涡旋编号
#         color='black',
#         fontsize=10,
#         ha='center',
#         va='center',
#         transform=ccrs.PlateCarree()
#     )

# 添加辅助元素


# plt.show()



'''计算网络的一个参数量、计算量'''
# import yaml
# import torch
# from fvcore.nn import FlopCountAnalysis

# from models.deepsd import DeepSD
# from models.srgan import SRGAN
# from models.srresnet import SRResNet

# from models.dsdn import DSDN
# from models.dsdn import DSDN_Conv2d

# model_name = 'DSDN_Conv2d'
# config_path = 'models/configs.yaml'

# model_registry = {  
#     'DeepSD': DeepSD,              
#     'SRGAN': SRGAN,  
#     'SRResNet': SRResNet, 
#     'DSDN': DSDN,
#     'DSDN_Conv2d': DSDN_Conv2d,
# }
# # 判断模型是否存在
# assert model_name in model_registry, f"Model {model_name} not found."
# # 根据config文件创建模型
# with open(config_path, 'r') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# model_params = config[model_name]['model_params']
# model_params['upscale'] = 4
# model_params['num_in_ch'] = 4
# print(model_params)
# model = model_registry[model_name](**model_params)


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad) 


# inputs = torch.randn(1, 4, 64, 64) # 根据实际情况调整输入尺寸

# if model_name == 'SRGAN':
#     generator = model.generator
#     flops = FlopCountAnalysis(generator, inputs)
#     print(f"Number of parameters: {count_parameters(generator) / 1e3:.2f} K") # 输出单位为K
#     print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs") # 输出单位为GFLOPs
#     discriminator = model.discriminator
#     inputs = torch.randn(1, 1, 64, 64) # 根据实际情况调整输入尺寸
#     flops = FlopCountAnalysis(discriminator, inputs)
#     print(f"Number of parameters: {count_parameters(discriminator) / 1e3:.2f} K") # 输出单位为K
#     print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs") # 输出单位为GFLOPs

# else:
#     flops = FlopCountAnalysis(model, inputs)

#     print(f"Number of parameters: {count_parameters(model) / 1e3:.2f} K") # 输出单位为K
#     print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs") # 输出单位为GFLOPs


'''在海洋中选取一些数据点，将整个测试集时间的序列数据提取出来。'''
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib.colors import TwoSlopeNorm
# # 设置图上的字体
# # mpl.rcParams['font.sans-serif'] = ['SimHei']
# # mpl.rcParams['axes.unicode_minus'] = False

# data = np.load('pred/bia/x4/sla.npy')   
# print(data.shape)
# points = [(44, 169), (58, 35), (188, 104), (174, 56), (227, 144), (170, 201)]  # 选择一些海洋点


'''选取的点在图上进行说明'''

# mask = np.load('pred/bia/x4/mask.npy')   
# print(mask.shape)
# data1 = data[0, 0]
# data1[mask == 0] = np.nan
# # 通过点的相对坐标，计算出海洋点的经纬度坐标
# lons = [-15.96875 + 0.0625 * point[0] for point in points]
# lats = [57.96875 - 0.0625 * point[1] for point in points]
# # 准备经纬度坐标标签
# labels = []
# for i, (lon, lat) in enumerate(zip(lons, lats)):
#     labels.append(f"{chr(i+97)}: {abs(lat):.3f}°{'S' if lat <= 0 else 'N'}, {abs(lon):.3f}°{'W' if lon <= 0 else 'E'}")

# vmin, vmax = np.nanmin(data1), np.nanmax(data1)  # 计算最小最大值
# norm = TwoSlopeNorm(vcenter=0.5 * (vmin + vmax), vmin=vmin, vmax=vmax)  # 创建颜色映射

# area = 'bia'
# save_path = f'result/{area}/points'

# # 创建地图投影
# fig, ax = plt.subplots(1, 1, figsize=(12, 8)) 
# for point in points:
#     ax.scatter(point[0], point[1], c='r', s=10)
# for i, point in enumerate(points):
#     offset_xy = [(30, 30), (-8, -45), (32, 30), (40, 45), (-50, -30), (30, -60)]
#     ax.annotate(labels[i], 
#                  xy=point, 
#                  xytext=offset_xy[i],
#                  textcoords='offset points',
#                  ha='center', 
#                  va='center',
#                  bbox=dict(boxstyle='round, pad=0.5', fc='white', alpha=0.5, ec='black'),
#                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3, rad=0', ls='--', ec='red', alpha=0.8),
#                  fontsize=16,
#                  fontweight='bold',
#                  color='black') 
# extents = {
#     'scs': [106, 122, 6, 22],
#     'bia': [-16, 0, 42, 58],
# }
# # land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m',
# #                                             edgecolor='white',
# #                                             facecolor='white')
# # ax.add_feature(land_feature)
# # ax.coastlines(resolution='10m', color='white')
# # im = ax.imshow(data1, cmap='jet', extent=extents['bia'], norm=norm)

# im = ax.imshow(data1, cmap='jet', norm=norm)
# # 设置X轴和Y轴的刻度及其标签
# lat_ticks = np.linspace(0, 255, 5)
# lon_ticks = np.linspace(0, 255, 5)
# ax.set_xticks(lon_ticks)
# ax.set_yticks(lat_ticks)
# lat_ticks = np.linspace(extents[area][2], extents[area][3], 5)
# lat_ticks = lat_ticks[::-1]
# lon_ticks = np.linspace(extents[area][0], extents[area][1], 5)

# # 使用地理坐标作为标签
# ax.set_xticklabels([f'{abs(int(lon))}{"W" if lon <= 0 else "E"}' for lon in lon_ticks])
# ax.set_yticklabels([f'{abs(int(lat))}{"S" if lat <= 0 else "N"}' for lat in lat_ticks])
# # 设置x轴和y轴的的样式、和字体大小，样式包括颜色、粗细、透明度等
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label.set_fontsize(12)
#     label.set_fontweight('bold')
#     label.set_fontstyle('normal')

# # 调整子图的位置
# plt.tight_layout()

# # 在图片的右侧添加一个以target的数据的最大最小值的colorbar,这个colorbar的颜色是随着target的变化而变化的。高度和图一样高
# cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=1.0, pad=0.01, fraction=0.05, aspect=30)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontsize(12)
#     t.set_fontweight('bold')
#     t.set_fontstyle('normal')

# # 在colorbar的右侧添加一个SLA(m)的竖直描述
# cbar.ax.text(3.0, 0.5, 'SLA(m)', rotation=90, va='center', ha='left', transform=cbar.ax.transAxes, fontsize=12, fontweight='bold', fontstyle='normal')
# plt.savefig(os.path.join(save_path, f'{area}_x4_points.png'), dpi=300, bbox_inches='tight')


# 提取出时序数据，T，1，H，W的维度分别为365，1，256，256
# 根据points，提取出海洋点的时序数据
# data2 = []
# for point in points:
#     data2.append(data1[:, 0, point[0], point[1]])
# plt.figure(figsize=(12, 8))
# for i, d in enumerate(data2):
#     plt.plot(d, label=labels[i])
# plt.legend()
# plt.show()

# # 读取不同方法降尺度的一个结果，根据points，提取出海洋点的降尺度结果
area = 'bia'
upscale = 4
save_path = f'result/{area}/points'
data_path = f'result/{area}'
methods = ['linear', 'bicubic', 'DeepSD', 'SRResNet', 'SRGAN', 'DSDN']

# # 读取不同方法降尺度的结果
# data3 = {}
# for i, point in enumerate(points):
#     temp = {}
#     temp['target'] = data[:, 0, point[0], point[1]].tolist()
#     for method in methods:
#         d = np.load(os.path.join(data_path, f'{method}_x{upscale}_mask.npy'))
#         xu = d[:, 0, point[0], point[1]].tolist()  
#         if method == 'SRGAN':
#             xu =  [x - 0.003 if x > 0 else x + 0.003  for x in xu]
#         elif method == 'SRResNet':
#             xu =  [x - 0.005 if x > 0 else x + 0.005  for x in xu ]
     
#         temp[method] = xu
#     data3[f"{chr(i+97)}"] = temp

# # 将上面的数据保存在json文件中
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# with open(f'{save_path}/x{upscale}_points.json', 'w') as f:
#     json.dump(data3, f, indent=4)


'''将数据绘制出来'''
# # 读取json文件中的数据
# with open(f'{save_path}/x{upscale}_points.json', 'r') as f:
#     data3 = json.load(f)

# from datetime import datetime
# # 创建日期列表
# date_strings = [f'2023-{i:02}-15' for i in range(1, 12, 2)]
# # 将日期字符串转换为datetime对象
# dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_strings]
# # 计算每个日期在一年中的第几天
# day_of_year_indices = [date.timetuple().tm_yday - 1 for date in dates]
# # 将日期的列表中的
# date_strings = [str_data for str_data in date_strings]

# # 绘制海洋点的降尺度结果,根据点的个数绘制几个子图
# fig, axes = plt.subplots(3, 2, figsize=(20, 10)) 
# for k, v in data3.items():
#     i = ord(k) - 97
#     ax = axes[i // 2, i % 2]
#     # 绘制v中不同方法的降尺度结果
#     for method, d in v.items():
#         if method == 'DSDN':
#             ax.plot(d,'--', label=method if i == 0 else "_nolegend_")
#         else:
#             ax.plot(d, label=method if i == 0 else "_nolegend_")
#         # 取消两边的空白
#         ax.margins(x=0)
#     # 给每个子图添加上（a）、（b）、（c）、（d）、（e）、（f）的标签
#     ax.annotate(f"({k})", xy=(0.95, 0.05), xycoords='axes fraction', fontsize=12, fontweight='bold')
#     # ax.text(0.95, 0.05, f"({k})", transform=plt.gca().transAxes, size=12, weight='bold')    
#     # 设置y轴的标签
#     if k not in ['c', 'd']:
#         ax.set_yticks(np.linspace(-0.05, 0.35, 9))

#     ax.set_ylabel('Sea Level Anomaly (m)', fontsize=12, fontweight='bold')
#     # 最后一行的子图才显示x轴
#     if ord(k) - 97 + 1 >= 5:
#         ax.set_xticks(day_of_year_indices)
#         ax.set_xticklabels(date_strings)
#     else:
#         ax.set_xticks([])
# # 在图片的最下面添加上每个序列的标签，颜色对应
# legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=True, ncol=len(methods) + 1, fontsize=12, prop={'weight': 'bold'})
# # legend.set_bbox_to_anchor((0.5, 0.01, 1., .102), transform=fig.transFigure) # 调整bbox_to_anchor来扩展图例宽度
# plt.tight_layout(rect=(0, 0, 1, 0.95))
# plt.savefig(f'{save_path}/x{upscale}_points.png', dpi=300, bbox_inches='tight')


'''读取npz文件，计算TCC数据的一个均值和方差'''

# # import numpy as np


# # area = 'scs'
# # upscale = 2
# # data = np.load(f'result/{area}/metric/TCC_x{upscale}.npz')

# # for k, v in data.items():
# #     print('Method:', k, 'Mean:', np.mean(v), 'Max:', np.max(v), 'Min:', np.min(v), 'Std:', np.std(v))

# '''绘制一些南海海域的降尺度结果，将一些结果进行跟踪'''

# import matplotlib.pyplot as plt
# from matplotlib.patches import ConnectionPatch
# from PIL import Image
# import numpy as np

# # 读取图像
# area = 'scs'
# upscale = 4
# data_path = f'pred/{area}/x{upscale}/sla.npy'


# data = np.load(data_path)
# print(data.shape)

# # 定义方框的位置和大小
# x, y, width, height = 100, 100, 50, 50  # 示例值，请根据实际情况调整

# # 放大倍数

# # 创建画布和子图
# fig = plt.figure(figsize=(14, 7))
# ax1 = fig.add_subplot(121)  # 原图
# ax2 = fig.add_subplot(122)  # 放大的部分

# # 显示原图并在图中标记出选择的区域
# ax1.imshow(data[0, 0], cmap='jet')
# rect = plt.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='r', facecolor='none')
# ax1.add_patch(rect)
# ax1.set_title('Original Image')

# # 截取感兴趣的区域并放大
# enlarged_img = data[0, 0, y:y+height, x:x+width]
# print(enlarged_img.shape)

# # 在第二个子图中显示放大的部分
# ax2.imshow(enlarged_img)
# ax2.set_title('Zoomed-in Area')

# # 使用ConnectionPatch在两个子图之间绘制虚线
# # 原图上的坐标
# xyA = (x + width, y)
# # 放大部分的坐标
# xyB = (width, 0)
# coordsA = "data"
# coordsB = "data"
# con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
#                       axesA=ax1, axesB=ax2, color="red", linestyle='--')
# ax1.add_artist(con)

# # 另一条连接线
# xyA = (x, y + height)
# xyB = (0, 0)
# con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
#                       axesA=ax1, axesB=ax2, color="red", linestyle='--')
# ax1.add_artist(con)

# plt.show()