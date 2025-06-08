import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SeaIceDataset(Dataset):
    """海冰数据集类"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        """
        Args:
            features: 输入特征 (N, H, W, C) 其中C=6个数据源
            targets: 目标值 (N, H, W, 1)
            transform: 数据变换
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, target

class SeaIceFusionCNN(nn.Module):
    """海冰密度融合CNN模型"""
    
    def __init__(self, in_channels=6, base_channels=64):
        super(SeaIceFusionCNN, self).__init__()
        
        # 编码器部分 - 提取特征
        self.encoder = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第二层卷积块
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            # 第三层卷积块
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # 解码器部分 - 重建输出
        self.decoder = nn.Sequential(
            # 上采样和卷积
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(base_channels//2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1之间的值
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 残差连接 - 将输入的平均值加到输出上
        residual = torch.mean(x, dim=1, keepdim=True)  # 对6个通道求平均
        residual = self.residual_conv(x)
        
        # 最终输出
        output = decoded + 0.1 * residual  # 加权残差连接
        output = torch.clamp(output, 0, 1)  # 确保输出在[0,1]范围内
        
        return output

class SeaIceDataLoader:
    """海冰数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.feature_names = ['OSI408', 'Bremen_ASI', 'NSIDC_NT2', 
                             'NSIDC_BT', 'NSIDC_NT', 'OSI401']
        self.target_name = 'Bremen_MODIS'
        
    def load_mat_file(self, filepath: str) -> Dict:
        """加载.mat文件（支持v7.3格式）"""
        try:
            # 首先尝试使用scipy.io.loadmat
            data = scipy.io.loadmat(filepath)
            # 移除MATLAB的元数据
            data = {k: v for k, v in data.items() if not k.startswith('__')}
            return data
        except NotImplementedError:
            # 如果是v7.3格式，使用h5py
            print(f"使用h5py加载MATLAB v7.3文件: {filepath}")
            try:
                data = {}
                with h5py.File(filepath, 'r') as f:
                    for key in f.keys():
                        if not key.startswith('#'):  # 跳过HDF5元数据
                            # 读取数据并转置（MATLAB和Python的数组顺序不同）
                            data[key] = np.array(f[key]).T
                return data
            except Exception as e:
                print(f"Error loading with h5py {filepath}: {e}")
                return {}
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def preprocess_data(self, data_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据"""
        print("开始数据预处理...")
        
        # 获取目标数据
        if self.target_name not in data_dict:
            raise ValueError(f"Target {self.target_name} not found in data")
        
        target = data_dict[self.target_name].astype(np.float32)
        
        # 获取特征数据
        features = []
        for name in self.feature_names:
            if name in data_dict:
                feature_data = data_dict[name].astype(np.float32)
                features.append(feature_data)
            else:
                print(f"Warning: {name} not found in data")
        
        if len(features) == 0:
            raise ValueError("No valid features found")
        
        features = np.stack(features, axis=-1)  # (H, W, T, N_features)
        
        print(f"原始数据形状 - Features: {features.shape}, Target: {target.shape}")
        
        # 数据清理
        features, target = self.clean_data(features, target)
        
        # 归一化到[0,1]
        features = np.clip(features / 100.0, 0, 1)
        target = np.clip(target / 100.0, 0, 1)
        
        # 重新组织数据维度为 (T, H, W, C) 用于CNN
        if len(features.shape) == 4:  # (H, W, T, C)
            features = np.transpose(features, (2, 3, 0, 1))  # (T, C, H, W)
            target = np.transpose(target, (2, 0, 1))  # (T, H, W)
            target = np.expand_dims(target, axis=1)  # (T, 1, H, W)
        
        print(f"预处理后数据形状 - Features: {features.shape}, Target: {target.shape}")
        
        return features, target
    
    def clean_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据清理"""
        print("进行数据清理...")
        
        # 处理异常值
        features[features < 0] = np.nan
        features[features > 100] = np.nan
        target[target < 0] = np.nan
        target[target > 100] = np.nan
        
        # 统计缺失值
        feature_nan_ratio = np.isnan(features).mean()
        target_nan_ratio = np.isnan(target).mean()
        
        print(f"特征数据缺失值比例: {feature_nan_ratio:.3f}")
        print(f"目标数据缺失值比例: {target_nan_ratio:.3f}")
        
        # 简单的缺失值填充 - 用邻近值填充
        if feature_nan_ratio < 0.5:  # 如果缺失值不太多
            features = self.fill_missing_values(features)
        if target_nan_ratio < 0.5:
            target = self.fill_missing_values(target)
        
        return features, target
    
    def fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        # 对于每个时间步，用空间邻近值填充
        for t in range(data.shape[-1] if len(data.shape) == 4 else data.shape[2]):
            if len(data.shape) == 4:  # features
                for c in range(data.shape[-1]):
                    slice_data = data[:, :, t, c]
                    if np.isnan(slice_data).any():
                        # 用均值填充
                        mean_val = np.nanmean(slice_data)
                        if not np.isnan(mean_val):
                            data[:, :, t, c] = np.where(np.isnan(slice_data), mean_val, slice_data)
            else:  # target
                slice_data = data[:, :, t]
                if np.isnan(slice_data).any():
                    mean_val = np.nanmean(slice_data)
                    if not np.isnan(mean_val):
                        data[:, :, t] = np.where(np.isnan(slice_data), mean_val, slice_data)
        
        return data
    
    def load_dataset(self, use_80day=True) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集"""
        subdir = "80day" if use_80day else "191day"
        data_path = os.path.join(self.data_dir, subdir)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory {data_path} not found")
        
        print(f"从 {data_path} 加载数据...")
        
        all_features = []
        all_targets = []
        
        # 加载所有.mat文件
        mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
        mat_files.sort()
        
        for mat_file in mat_files:
            filepath = os.path.join(data_path, mat_file)
            print(f"加载文件: {mat_file}")
            
            data_dict = self.load_mat_file(filepath)
            if data_dict:
                try:
                    features, target = self.preprocess_data(data_dict)
                    all_features.append(features)
                    all_targets.append(target)
                except Exception as e:
                    print(f"处理文件 {mat_file} 时出错: {e}")
                    continue
        
        if not all_features:
            raise ValueError("No valid data loaded")
        
        # 合并所有数据
        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        print(f"最终数据形状 - Features: {features.shape}, Targets: {targets.shape}")
        
        return features, targets

class SeaIceTrainer:
    """海冰融合模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, lr: float = 0.001):
        """训练模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        
        print(f"开始训练，设备: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_sea_ice_model.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 展平数据计算指标
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # 移除NaN值
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 设置参数
    DATA_DIR = "assets"
    BATCH_SIZE = 8  # 根据GPU内存调整
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print("=== 海冰密度融合CNN系统 ===")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_loader = SeaIceDataLoader(DATA_DIR)
    
    try:
        # 先尝试加载80天高质量数据
        features, targets = data_loader.load_dataset(use_80day=True)
    except Exception as e:
        print(f"加载80天数据失败: {e}")
        print("尝试加载191天数据...")
        features, targets = data_loader.load_dataset(use_80day=False)
    
    # 2. 数据分割
    print("\n2. 数据分割...")
    n_samples = features.shape[0]
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_features = features[train_indices]
    train_targets = targets[train_indices]
    val_features = features[val_indices]
    val_targets = targets[val_indices]
    test_features = features[test_indices]
    test_targets = targets[test_indices]
    
    print(f"训练集: {train_features.shape[0]} 样本")
    print(f"验证集: {val_features.shape[0]} 样本")
    print(f"测试集: {test_features.shape[0]} 样本")
    
    # 3. 创建数据加载器
    print("\n3. 创建数据加载器...")
    train_dataset = SeaIceDataset(train_features, train_targets)
    val_dataset = SeaIceDataset(val_features, val_targets)
    test_dataset = SeaIceDataset(test_features, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 创建模型
    print("\n4. 创建模型...")
    model = SeaIceFusionCNN(in_channels=6, base_channels=64)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 5. 训练模型
    print("\n5. 开始训练...")
    trainer = SeaIceTrainer(model)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    
    # 6. 评估模型
    print("\n6. 评估模型...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_sea_ice_model.pth'))
    trainer.model = model
    
    # 在测试集上评估
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n=== 测试集评估结果 ===")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 7. 绘制训练历史
    print("\n7. 绘制训练历史...")
    trainer.plot_training_history()
    
    print("\n=== 训练完成 ===")
    print("最佳模型已保存为: best_sea_ice_model.pth")
    print("训练历史图已保存为: training_history.png")

if __name__ == "__main__":
    main() 