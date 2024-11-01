# model.py

import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        """
        初始化 AlexNet 模型。

        :param num_classes: 分类的类别数，默认为1000
        :param init_weights: 是否初始化权重，默认为False
        """
        super(AlexNet, self).__init__()  # 调用父类的初始化方法

        # 定义特征提取层
        self.features = nn.Sequential(  # 使用 Sequential 包装多个层
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # 输入 [3, 224, 224]，输出 [48, 55, 55]
            nn.ReLU(inplace=True),  # ReLU 激活函数，inplace=True 可以节省内存
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化，输出 [48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # 输出 [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出 [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 输出 [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 输出 [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [128, 6, 6]
        )

        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout 层，防止过拟合
            nn.Linear(128 * 6 * 6, 2048),  # 全连接层，输入 [128 * 6 * 6]，输出 [2048]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 全连接层，输入 [2048]，输出 [2048]
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # 全连接层，输入 [2048]，输出 [num_classes]
        )

        # 如果需要初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入张量
        :return: 输出张量
        """
        x = self.features(x)  # 通过特征提取层
        x = torch.flatten(x, start_dim=1)  # 将特征图展平，start_dim=1 表示从第1维开始展平
        x = self.classifier(x)  # 通过分类器
        return x

    def _initialize_weights(self):
        """
        初始化模型权重。
        """
        for m in self.modules():  # 遍历所有模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用 Kaiming 初始化方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置项初始化为0
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 使用正态分布初始化权重
                nn.init.constant_(m.bias, 0)  # 偏置项初始化为0
