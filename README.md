# 海冰密度融合CNN系统

基于卷积神经网络的多源海冰密集度数据融合系统，用于提高海冰密集度预测精度。

## 项目概述

本项目实现了一个端到端的海冰密集度数据融合系统，使用CNN模型融合6个不同数据源的海冰密集度数据：

- **输入数据源（6个）**：
  - OSI408 (OSISAF AMSR2产品)
  - Bremen_ASI (不来梅大学ASI算法)
  - NSIDC_NT2 (NASA Team 2算法)
  - NSIDC_BT (Bootstrap算法)
  - NSIDC_NT (NASA Team算法)
  - OSI401 (OSISAF SSMIS产品)

- **目标数据**：Bremen_MODIS (光学遥感高精度数据)

## 项目结构

```
├── sea_ice_fusion_cnn.py      # 主要CNN融合系统代码
├── demo_cnn_fusion.py         # 简化演示程序（推荐使用）
├── generate_synthetic_data.py # 合成数据生成器
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目说明
├── assets/                    # 原始数据目录
│   ├── 80day/                 # 80天高质量数据
│   └── 191day/                # 191天大容量数据
├── synthetic_data/            # 合成数据目录
│   ├── 80day/                 # 合成的80天数据
│   └── 191day/                # 合成的191天数据
└── results/                   # 输出结果目录
    ├── *.png                  # 生成的图表
    └── *.pth                  # 训练好的模型
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐使用uv）
uv venv
source .venv/bin/activate

# 或使用传统方式
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示程序（推荐）

如果你想快速体验CNN融合系统的效果：

```bash
# 运行简化演示程序
python demo_cnn_fusion.py
```

这个演示程序会：
- 自动生成合成的海冰数据
- 训练一个简化的CNN模型
- 展示融合效果和评估结果
- 生成可视化图表

### 3. 使用真实数据训练

如果你有真实的海冰数据文件：

```bash
# 运行完整的CNN融合系统
python sea_ice_fusion_cnn.py
```

### 4. 生成合成数据

如果需要生成更多的合成数据用于测试：

```bash
# 生成合成海冰数据集
python generate_synthetic_data.py
```

## 详细使用说明

### 数据格式要求

- **文件格式**：MATLAB .mat文件（支持v7.3格式）
- **数据结构**：
  ```
  数据字段：
  - Bremen_MODIS: (H, W, T) - 目标数据
  - OSI408: (H, W, T) - 输入数据源1
  - Bremen_ASI: (H, W, T) - 输入数据源2
  - NSIDC_NT2: (H, W, T) - 输入数据源3
  - NSIDC_BT: (H, W, T) - 输入数据源4
  - NSIDC_NT: (H, W, T) - 输入数据源5
  - OSI401: (H, W, T) - 输入数据源6
  
  其中：H=高度, W=宽度, T=时间步数
  ```
- **数值范围**：0-100（海冰密集度百分比）
- **缺失值**：使用NaN表示陆地或无效区域

### 模型配置

可以在代码中修改以下参数：

```python
# 在sea_ice_fusion_cnn.py的main()函数中
DATA_DIR = "assets"           # 数据目录
BATCH_SIZE = 8               # 批次大小（根据GPU内存调整）
NUM_EPOCHS = 50              # 训练轮数
LEARNING_RATE = 0.001        # 学习率

# 在demo_cnn_fusion.py中
BATCH_SIZE = 4               # 演示用较小批次
NUM_EPOCHS = 20              # 演示用较少轮数
```

### 模型架构

#### 完整CNN模型 (SeaIceFusionCNN)
- **编码器**：3层卷积块，逐步提取特征
- **解码器**：3层反卷积，重建输出
- **残差连接**：改善梯度流动
- **输出激活**：Sigmoid确保输出在[0,1]范围

#### 简化CNN模型 (SimpleSeaIceCNN)
- **4层卷积**：32→64→32→1通道
- **批归一化**：提高训练稳定性
- **ReLU激活**：非线性变换
- **Sigmoid输出**：归一化输出

## 运行流程详解

### 方案一：快速演示（推荐新手）

1. **环境准备**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **运行演示**
   ```bash
   python demo_cnn_fusion.py
   ```

3. **查看结果**
   - 训练过程会在终端显示
   - 生成的图表保存在 `results/` 目录
   - 包括训练历史和预测结果可视化

### 方案二：完整训练流程

1. **数据准备**
   - 将.mat文件放入 `assets/80day/` 或 `assets/191day/`
   - 确保数据格式符合要求

2. **运行训练**
   ```bash
   python sea_ice_fusion_cnn.py
   ```

3. **训练过程**
   - 自动加载和预处理数据
   - 数据分割（70%训练，15%验证，15%测试）
   - CNN模型训练和验证
   - 最佳模型保存

4. **结果输出**
   - `best_sea_ice_model.pth`：最佳模型权重
   - `training_history.png`：训练历史图表
   - 终端显示评估指标（RMSE, MAE, R²）

### 方案三：自定义数据生成

1. **生成合成数据**
   ```bash
   python generate_synthetic_data.py
   ```

2. **修改数据参数**
   ```python
   # 在generate_synthetic_data.py中修改
   height=128, width=128, time_steps=20  # 数据尺寸
   ```

3. **使用生成的数据**
   - 数据保存在 `synthetic_data/` 目录
   - 可用于测试和算法验证

## 评估指标

- **RMSE**：均方根误差（越小越好）
- **MAE**：平均绝对误差（越小越好）
- **R²**：决定系数（越接近1越好）

## 输出文件说明

### 模型文件
- `best_sea_ice_model.pth`：训练好的最佳模型权重

### 可视化文件
- `training_history.png`：训练和验证损失曲线
- `simple_training_history.png`：简化模型训练历史
- `prediction_results.png`：预测结果对比图
- `simple_prediction_results.png`：简化模型预测结果
- `synthetic_data_preview.png`：合成数据预览

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 解决方案：减小批次大小
   BATCH_SIZE = 2  # 或更小
   ```

2. **CUDA错误**
   ```bash
   # 解决方案：强制使用CPU
   device = torch.device('cpu')
   ```

3. **数据加载失败**
   ```bash
   # 解决方案：使用演示程序
   python demo_cnn_fusion.py
   ```

4. **训练损失为NaN**
   ```bash
   # 解决方案：使用简化演示程序
   python demo_cnn_fusion.py
   ```

### 性能优化

1. **GPU加速**
   - 确保安装CUDA版本的PyTorch
   - 增大批次大小以充分利用GPU

2. **训练速度**
   - 减少数据尺寸
   - 使用较少的训练轮数进行快速测试

3. **模型精度**
   - 增加训练轮数
   - 调整学习率
   - 尝试不同的模型架构

## 技术特点

### 数据处理
- 自动检测MATLAB文件格式（v7.3支持）
- 智能缺失值处理
- 数据归一化和维度调整
- 支持多文件批量处理

### 模型设计
- 编码器-解码器架构
- 残差连接机制
- 批归一化稳定训练
- 梯度裁剪防止爆炸

### 训练策略
- 自适应学习率调度
- 早停机制防止过拟合
- 模型检查点保存
- 完整的评估指标

## 扩展功能

可以考虑的改进方向：

1. **注意力机制**：加入空间注意力模块
2. **时序建模**：考虑时间序列特征
3. **多尺度融合**：处理不同分辨率数据
4. **不确定性估计**：输出预测置信度
5. **在线学习**：支持增量训练

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

## 许可证

本项目采用MIT许可证。
