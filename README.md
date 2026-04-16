# RM_CNN_TRAINING

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Kielas520/RM_CNN_TRAINING.git)

本项目是一个专为 RoboMaster (RM) 视觉系统设计的装甲板数字与类别识别卷积神经网络（CNN）训练框架。该框架提供了一套完整的端到端工作流，涵盖数据集自动化预处理、数据平衡、模型训练与评估（基于 PyTorch）、以及最终导出为适用于 C++ 部署的 ONNX 模型。

## 环境安装

本项目使用现代化的 Python 构建系统，推荐使用 `uv` 或标准的 `pip` 进行依赖管理。支持的 Python 版本为 `3.10` 且 `<3.11`。

```bash
# 克隆仓库
git clone https://github.com/Kielas520/RM_CNN_TRAINING.git
cd RM_CNN_TRAINING

# 安装核心训练依赖（包含 PyTorch CUDA 12.4 支持）
uv pip install -e .

# 安装部署相关的额外依赖（用于导出 ONNX）
uv pip install -e ".[deploy]"
```

## 数据集准备与采集

高质量的分类模型高度依赖于规范的数据采集。请按照以下步骤构建你的 `raw` 数据集。

### 1. 负样本（Negative）准备

本项目使用 CIFAR-100 作为非装甲板（负样本）的背景数据来源。
1. 前往 [CIFAR-100 官网](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 Python 版本数据集并解压至项目根目录。
2. 运行预处理脚本，这会自动将 CIFAR-100 转换为 20x28 尺寸的二值化图片，并输出至对应目录：
   ```bash
   python src/process_cifra100.py
   ```

### 2. 装甲板正样本数据采集 (ROS 2)

正样本需要通过真实相机与识别器在不同工况下录制 ROS 2 Bag 获得。

1. **启动节点**：启动你的相机节点与装甲板识别节点。
2. **位置校验**：将装甲板置于相机视野中，拉远到期望的极限识别距离，检查当前识别器提取到的装甲板四个角点是否仍然准确。
3. **录制数据**：在角点提取准确的前提下，改变装甲板的姿态（俯仰、偏航），针对不同距离录制对应类别的 rosbag。录制的主题为二值化处理后的装甲板 ROI：
   ```bash
   ros2 bag record /detector/img_armor_processed -o armor_bag1.1
   ```
   *建议的录制规范（以英雄/步兵/哨兵为例）：*
   * `bag1.1` -> 近距离 (Close)
   * `bag1.2` -> 中距离 (Middle)
   * `bag1.3` -> 远距离 (Far)

### 3. 数据提取与目录结构

使用提供的脚本从录制好的 ROS 2 bag 中提取二值图文件：

```bash
# 示例：将基础名为 armor_bag1 (包含 armor_bag1.1, armor_bag1.2 等) 的包提取到 1 号文件夹
python src/extract_bag_bin.py ./armor_bag1 ./data/raw/1/
```

最终提取出的图片，请严格按照以下结构放置在 `data/raw` 文件夹下。分类映射可在 `config.yaml` 中自定义：

```text
data/raw/
 ├─ 1/           # 类别 0 (例如: 英雄 / 数字 1)
 ├─ 2/           # 类别 1 (例如: 步兵 / 数字 3)
 ├─ 3/           # 类别 2 (例如: 哨兵)
 └─ 4negative/   # 类别 3 (非装甲板 / 啥也不是)
```

## 训练与部署工作流

所有的核心参数（学习率、Batch Size、网络输入尺寸、数据增强策略等）均在 `config.yaml` 中配置。

### 1. 验证数据流与增强效果

在开始训练前，可以独立运行可视化脚本。该脚本会自动触发 `data/raw` 到 `data/processed` 的尺寸规范化和样本平衡处理，并预览数据增强效果：

```bash
python src/test_data_viz.py
```

### 2. 模型训练

执行训练脚本。训练过程中会自动进行验证集评估，并在 `deploy` 目录下保存最佳模型权重 (`_best.pth`)。

```bash
python src/train.py
```

**查看训练曲线**：
本框架支持 TensorBoard 监控，可通过以下命令查看 Loss、Accuracy 和学习率衰减曲线：
```bash
tensorboard --logdir=runs
```

### 3. 模型导出 (ONNX)

训练彻底结束后，使用导出脚本将 PyTorch 模型转换为 C++ 端可加载的 ONNX 格式。该脚本会自动执行：
1. 导出原始 `.onnx`
2. 消除冗余算子生成简化版 `_sim.onnx`（推荐在 C++ 部署时首选此版本）
3. 执行 INT8 动态量化生成 `_quant.onnx`（可用于极致压缩模型体积）

```bash
python src/export.py
```
