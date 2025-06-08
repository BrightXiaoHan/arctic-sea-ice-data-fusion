"""
海冰密度融合CNN系统配置文件
"""

import os

# 数据配置
DATA_CONFIG = {
    'data_dir': 'assets',
    'synthetic_data_dir': 'synthetic_data',
    'results_dir': 'results',
    'feature_names': ['OSI408', 'Bremen_ASI', 'NSIDC_NT2', 'NSIDC_BT', 'NSIDC_NT', 'OSI401'],
    'target_name': 'Bremen_MODIS',
    'use_80day': True,  # True: 使用80天数据, False: 使用191天数据
}

# 模型配置
MODEL_CONFIG = {
    'in_channels': 6,
    'base_channels': 64,  # 完整模型使用64，演示模型使用32
    'input_size': (128, 128),  # (height, width)
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# 演示配置（用于快速测试）
DEMO_CONFIG = {
    'batch_size': 4,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'model_channels': 32,
    'data_size': (64, 64),  # 较小的数据尺寸
}

# 输出配置
OUTPUT_CONFIG = {
    'model_save_path': 'best_sea_ice_model.pth',
    'training_history_path': 'results/training_history.png',
    'prediction_results_path': 'results/prediction_results.png',
    'synthetic_preview_path': 'results/synthetic_data_preview.png',
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,  # 是否使用CUDA
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
}

def get_device():
    """获取计算设备"""
    import torch
    
    if DEVICE_CONFIG['device'] == 'auto':
        if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(DEVICE_CONFIG['device'])

def create_directories():
    """创建必要的目录"""
    directories = [
        DATA_CONFIG['results_dir'],
        os.path.join(DATA_CONFIG['synthetic_data_dir'], '80day'),
        os.path.join(DATA_CONFIG['synthetic_data_dir'], '191day'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 在导入时自动创建目录
create_directories() 