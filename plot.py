import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')  # 关闭坐标轴

# 定义绘制矩形框的函数
def draw_box(ax, x, y, width, height, text, color='lightblue'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10)

# 定义绘制箭头的函数
def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20, color='black')
    ax.add_patch(arrow)

# 1. 输入
ax.text(0.5, 7.5, 'LR Image', fontsize=12)
draw_arrow(ax, (1, 7.5), (2, 7))

# 2. 浅层特征提取
draw_box(ax, 2, 6.5, 2, 1, 'BSConv (3×3)')
draw_arrow(ax, (4, 7), (5, 7))

# 3. 深层特征提取 (多个 BSRB + 注意力模块)
for i in range(2):  # 示例：绘制 2 个 BSRB 单元
    x_offset = 5 + i * 2.5
    # BSRB
    draw_box(ax, x_offset, 6, 1, 0.8, 'BSConv (1×1)', color='lightgreen')
    draw_box(ax, x_offset + 1, 6, 1, 0.8, 'BSConv (3×3)', color='lightgreen')
    ax.text(x_offset + 1, 7, f'BSRB {i+1}', ha='center', fontsize=10)
    draw_arrow(ax, (x_offset + 0.5, 6.8), (x_offset + 1, 6.8))  # 内部连接
    draw_arrow(ax, (x_offset - 0.5, 6.4), (x_offset + 2, 6.4))  # 残差连接
    ax.text(x_offset + 2, 6.4, '+', fontsize=12)
    # 注意力模块
    draw_box(ax, x_offset + 0.5, 5, 1, 0.8, 'CCA/ESA', color='lightyellow')
    draw_arrow(ax, (x_offset + 1, 6), (x_offset + 1, 5.8))

# 连接深层特征提取部分
draw_arrow(ax, (4, 6.5), (5, 6.5))
draw_arrow(ax, (7, 5.5), (8, 6))

# 4. 图像重建
draw_box(ax, 8, 6, 1, 0.8, 'BSConv (1×1)')
draw_box(ax, 9, 6, 1, 0.8, 'PixelShuffle')
draw_box(ax, 10, 6, 1, 0.8, 'BSConv (3×3)')
draw_arrow(ax, (9, 6.4), (10, 6.4))
draw_arrow(ax, (10, 6.4), (11, 6.4))
ax.text(11, 6.4, '+', fontsize=12)

# 5. 全局残差连接
draw_arrow(ax, (4, 6.8), (8, 7.2))  # 从浅层特征到重建
draw_arrow(ax, (8, 7.2), (11, 6.8))

# 6. 输出
ax.text(11.5, 6.5, 'HR Image', fontsize=12)

# 显示图像
plt.title('BSRN Network Structure', fontsize=14)
plt.show()