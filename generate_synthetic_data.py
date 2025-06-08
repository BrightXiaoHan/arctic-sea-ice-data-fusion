import numpy as np
import scipy.io
import os
from typing import Tuple
import matplotlib.pyplot as plt

def generate_synthetic_sea_ice_data(height: int = 256, width: int = 256, 
                                   time_steps: int = 20) -> dict:
    """
    生成合成的海冰密集度数据
    
    Args:
        height: 图像高度
        width: 图像宽度  
        time_steps: 时间步数
        
    Returns:
        包含所有数据源的字典
    """
    print(f"生成合成数据: {height}x{width}x{time_steps}")
    
    # 创建基础的海冰分布模式
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 基础海冰分布（极地模式）
    base_ice = np.exp(-(X**2 + Y**2) / 0.5) * 100  # 中心高密度
    
    # 添加一些复杂的地理特征
    # 海岸线效应
    coastline = 20 * np.sin(5 * X) * np.cos(3 * Y)
    
    # 洋流效应
    current_effect = 15 * np.sin(2 * np.pi * (X + Y))
    
    data_dict = {}
    
    # 生成真实数据 (Bremen_MODIS) - 最高精度
    bremen_modis = np.zeros((height, width, time_steps))
    for t in range(time_steps):
        # 时间变化：季节性变化
        seasonal_factor = 0.3 * np.sin(2 * np.pi * t / time_steps)
        
        # 随机天气扰动
        weather_noise = np.random.normal(0, 5, (height, width))
        
        # 组合所有效应
        ice_concentration = (base_ice + coastline + current_effect + 
                           seasonal_factor * 20 + weather_noise)
        
        # 限制在0-100范围内
        ice_concentration = np.clip(ice_concentration, 0, 100)
        
        # 添加一些陆地区域（设为NaN）
        land_mask = (X**2 + Y**2) > 0.8
        ice_concentration[land_mask] = np.nan
        
        bremen_modis[:, :, t] = ice_concentration
    
    data_dict['Bremen_MODIS'] = bremen_modis
    
    # 生成6个输入数据源，每个都有不同的特点和误差
    feature_names = ['OSI408', 'Bremen_ASI', 'NSIDC_NT2', 'NSIDC_BT', 'NSIDC_NT', 'OSI401']
    
    for i, name in enumerate(feature_names):
        feature_data = np.zeros((height, width, time_steps))
        
        for t in range(time_steps):
            # 基于真实数据，但添加不同的偏差和噪声
            true_data = bremen_modis[:, :, t].copy()
            
            # 不同数据源的特定偏差
            if 'OSI' in name:
                # OSI产品：系统性低估
                bias = -5 - 2 * i
                noise_level = 3 + i
            elif 'NSIDC' in name:
                # NSIDC产品：不同的系统偏差
                bias = (-1)**i * (3 + i)
                noise_level = 4 + i * 0.5
            else:
                # Bremen_ASI：高分辨率但有噪声
                bias = 2
                noise_level = 6
            
            # 添加偏差和噪声
            noisy_data = true_data + bias + np.random.normal(0, noise_level, true_data.shape)
            
            # 添加一些系统性的空间误差
            spatial_error = 3 * np.sin(i * X) * np.cos(i * Y)
            noisy_data += spatial_error
            
            # 限制范围
            noisy_data = np.clip(noisy_data, 0, 100)
            
            # 保持陆地掩码
            noisy_data[np.isnan(true_data)] = np.nan
            
            feature_data[:, :, t] = noisy_data
        
        data_dict[name] = feature_data
    
    # 添加坐标信息
    data_dict['modis_x'] = x.reshape(-1, 1)
    data_dict['modis_y'] = y.reshape(-1, 1)
    
    # 添加时间信息
    data_dict['selected_dates'] = np.arange(time_steps).reshape(1, -1)
    
    return data_dict

def save_synthetic_data(data_dict: dict, filepath: str):
    """保存合成数据为.mat文件"""
    print(f"保存数据到: {filepath}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为MATLAB格式
    scipy.io.savemat(filepath, data_dict)

def visualize_synthetic_data(data_dict: dict, save_path: str = "synthetic_data_preview.png"):
    """可视化合成数据"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 显示真实数据和6个输入源的第一个时间片
    keys = ['Bremen_MODIS', 'OSI408', 'Bremen_ASI', 'NSIDC_NT2', 
            'NSIDC_BT', 'NSIDC_NT', 'OSI401']
    
    for i, key in enumerate(keys):
        if i < len(axes) and key in data_dict:
            data_slice = data_dict[key][:, :, 0]  # 第一个时间片
            
            im = axes[i].imshow(data_slice, cmap='Blues', vmin=0, vmax=100)
            axes[i].set_title(f'{key}\n(时间片 0)')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # 隐藏最后一个子图
    if len(keys) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化图像已保存: {save_path}")

def create_synthetic_datasets():
    """创建完整的合成数据集"""
    print("=== 创建合成海冰数据集 ===")
    
    # 创建80天数据集（高质量，小尺寸）
    print("\n创建80天数据集...")
    os.makedirs("synthetic_data/80day", exist_ok=True)
    
    for i in range(3):  # 创建3个文件
        data_dict = generate_synthetic_sea_ice_data(
            height=128, width=128, time_steps=20
        )
        filepath = f"synthetic_data/80day/sea_ice_dataset{i+1}.mat"
        save_synthetic_data(data_dict, filepath)
    
    # 创建191天数据集（大容量）
    print("\n创建191天数据集...")
    os.makedirs("synthetic_data/191day", exist_ok=True)
    
    for i in range(2):  # 创建2个文件
        data_dict = generate_synthetic_sea_ice_data(
            height=128, width=128, time_steps=40
        )
        filepath = f"synthetic_data/191day/sea_ice_dataset{i+1}.mat"
        save_synthetic_data(data_dict, filepath)
    
    # 可视化一个样本
    print("\n生成可视化...")
    sample_data = generate_synthetic_sea_ice_data(height=128, width=128, time_steps=5)
    visualize_synthetic_data(sample_data)
    
    print("\n=== 合成数据集创建完成 ===")
    print("数据保存在 synthetic_data/ 目录中")
    print("可以使用这些数据来测试CNN融合系统")

if __name__ == "__main__":
    create_synthetic_datasets() 