### README.md

# AlexNet 模型实现

## 项目概述

这是一个基于 AlexNet模型的图像分类项目。该项目使用 PyTorch 实现，旨在对给定的图像进行分类。项目包含图像预处理、模型加载、预测和结果展示等功能。

## 目录结构
- `confusion_matrix/`: 存放每次训练的所有轮次混淆矩阵图。
- `data/`: 数据集目录。
  - `flower_photos/`: 解压后的图像文件。
  - `train/`: 训练集图像。
  - `val/`: 验证集图像。
  - `flower_photos.tgz`: 未解压的图像文件。
- `models/`: 模型相关文件。
  - `train_logs/`: 训练日志图。
  - `model.py`: 模型定义。
  - `AlexNet.pth`: 预训练模型权重。
- `src/`: 脚本文件夹。
  - `check_gpu.py`: 检查 GPU 是否可用。
  - `class_indices.json`: 类索引文件。
  - `dataload.py`: 数据集下载脚本。
  - `predict.py`: 验证脚本。
  - `split_data.py`: 划分数据集脚本。
  - `train.py`: 训练脚本。
- `README.md`: 项目说明。
- `requirements.txt`: 依赖项列表。

## 环境配置

### 安装依赖项

请确保已安装以下依赖项。可以使用以下命令安装：

```sh
pip install -r requirements.txt
```

### 依赖项列表

- torch~=2.1.2+cu121

- matplotlib~=3.7.2
- seaborn~=0.13.2
- torchvision~=0.16.2+cu121
- scikit-learn~=1.3.2
- numpy~=1.24.4
- pillow~=10.4.0

## 运行项目

### 下载数据集

运行 `dataload.py` 脚本来对图像进行分类预测：

```sh
python src/dataload.py
```
### 划分数据集

运行 `split_data.py` 脚本来对图像进行分类预测：

```sh
python src/split_data.py
```
### 训练模型

运行 `train.py` 脚本来对图像进行分类预测：

```sh
python src/train.py
```
### 验证模型

运行 `predict.py` 脚本来对图像进行分类预测：

```sh
python src/predict.py
```
### 预测脚本

运行 `predict.py` 脚本来对图像进行分类预测：

```sh
python predict.py
```

## 贡献

欢迎贡献！如果你有任何改进建议或发现任何问题，请提交一个 Pull Request 或 Issue。

## 感谢
特别感谢以下贡献者：

@zhangyunjin488 - 提供了数据集下载脚本，审核了代码并提出了宝贵意见。
@lydia6618 - 帮助优化了模型训练代码，审核了代码并提出了宝贵意见。

## 许可证

本项目遵循 MIT 许可证，详情请参见 [LICENSE](LICENSE) 文件。

## 联系方式
如果有任何问题或建议，请联系 [jiawei7898@gmail.com]。