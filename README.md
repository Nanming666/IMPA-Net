# IMPA-Net
IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss for Extreme Convective Radar Nowcasting

**Code will be released upon publication.**
# IMPA 雷达回波临近预报模型

基于 IMPA 框架的雷达回波临近预报系统。

## 环境安装

```bash
# 创建 conda 环境
conda create -n impa python=3.9
conda activate impa

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

将数据放在指定路径，目录结构如下：
```
data/
├── Train/
│   └── Radar/         # 雷达回波图像
├── TestA/
│   └── Radar/
├── dataset_train.csv  # 训练集文件列表
└── dataset_testA.csv  # 测试集文件列表
```

## 训练

**单卡训练**：
```bash
bash scripts/train_single_gpu.sh
```

**多卡并行训练**：
```bash
bash scripts/train_multi_gpu.sh
```

### 训练配置

在 `configs/` 目录下选择合适的配置文件。

### 可调参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `hid_S` | 空间隐层维度 | 32 |
| `hid_T` | 时序隐层维度 | 256 |
| `N_T` | Translator层数 | 12 |
| `N_S` | 编解码器层数 | 8 |
| `lr` | 学习率 | 8e-4 |
| `batch_size` | 批大小 | 2 |
| `epoch` | 训练轮数 | 50 |

## 输入输出格式

- **输入**: `[B, T, C, H, W]` — 批次×时间步×通道×高×宽
- **输出**: `[B, T, 1, H, W]` — 未来雷达回波预测

## 微调

1. 准备自己的数据，按上述目录结构组织
2. 修改配置文件中的数据路径
3. 根据数据规模调整 `batch_size`、`lr`、`epoch` 等参数
4. 运行训练脚本

## 测试

```bash
python tools/test.py --config configs/your_config.py --ex_name your_experiment
```

## 可视化

```bash
python tools/visualizations/visualize_radar.py --data_path /path/to/data
```
