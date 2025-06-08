#!/usr/bin/env python3
"""
修复版海冰密度融合CNN演示程序
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleSeaIceCNN(nn.Module):
    """简化的海冰融合CNN模型"""
    
    def __init__(self, in_channels=6):
        super(SimpleSeaIceCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.features(x)

def generate_simple_data(batch_size=32, height=64, width=64, num_channels=6):
    """生成简单的测试数据"""
    # 创建有意义的合成数据
    features = torch.randn(batch_size, num_channels, height, width) * 0.1 + 0.5
    features = torch.clamp(features, 0, 1)
    
    # 目标是输入的加权平均，加上一些噪声
    weights = torch.tensor([0.2, 0.15, 0.15, 0.15, 0.15, 0.2]).view(1, 6, 1, 1)
    targets = torch.sum(features * weights, dim=1, keepdim=True)
    targets = targets + torch.randn_like(targets) * 0.05
    targets = torch.clamp(targets, 0, 1)
    
    return features, targets

def train_simple_model():
    """训练简化模型"""
    print("=== 简化海冰密度融合CNN演示 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleSeaIceCNN(in_channels=6).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 设置训练参数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 20
    train_losses = []
    
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 10
        
        for batch in range(num_batches):
            # 生成批次数据
            features, targets = generate_simple_data(batch_size=8, height=64, width=64)
            features = features.to(device)
            targets = targets.to(device)
            
            # 检查数据是否包含NaN
            if torch.isnan(features).any() or torch.isnan(targets).any():
                print(f"警告: 数据包含NaN值")
                continue
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"警告: 损失为NaN，跳过此批次")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # 测试模型
    print("\n测试模型...")
    model.eval()
    
    with torch.no_grad():
        # 生成测试数据
        test_features, test_targets = generate_simple_data(batch_size=4, height=64, width=64)
        test_features = test_features.to(device)
        test_targets = test_targets.to(device)
        
        # 预测
        predictions = model(test_features)
        
        # 计算测试损失
        test_loss = criterion(predictions, test_targets).item()
        print(f"测试损失: {test_loss:.6f}")
        
        # 计算评估指标
        pred_np = predictions.cpu().numpy().flatten()
        target_np = test_targets.cpu().numpy().flatten()
        
        # 移除可能的NaN值
        valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
        if valid_mask.sum() > 0:
            pred_valid = pred_np[valid_mask]
            target_valid = target_np[valid_mask]
            
            rmse = np.sqrt(mean_squared_error(target_valid, pred_valid))
            mae = mean_absolute_error(target_valid, pred_valid)
            r2 = r2_score(target_valid, pred_valid)
            
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.6f}")
        
        # 可视化结果
        visualize_results(test_features, test_targets, predictions)
    
    # 绘制训练历史
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 训练完成 ===")
    print("这个简化演示展示了CNN融合的基本原理")

def visualize_results(features, targets, predictions, num_samples=2):
    """可视化预测结果"""
    features_np = features[:num_samples].cpu().numpy()
    targets_np = targets[:num_samples].cpu().numpy()
    predictions_np = predictions[:num_samples].cpu().numpy()
    
    fig, axes = plt.subplots(num_samples, 8, figsize=(16, num_samples*2))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 显示6个输入通道
        for j in range(6):
            im = axes[i, j].imshow(features_np[i, j], cmap='Blues', vmin=0, vmax=1)
            axes[i, j].set_title(f'Input {j+1}')
            axes[i, j].axis('off')
        
        # 显示真实值
        im = axes[i, 6].imshow(targets_np[i, 0], cmap='Blues', vmin=0, vmax=1)
        axes[i, 6].set_title('Ground Truth')
        axes[i, 6].axis('off')
        
        # 显示预测值
        im = axes[i, 7].imshow(predictions_np[i, 0], cmap='Blues', vmin=0, vmax=1)
        axes[i, 7].set_title('Prediction')
        axes[i, 7].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("预测结果已保存: simple_prediction_results.png")

if __name__ == "__main__":
    train_simple_model() 