# predict.py
import os
import numpy as np
import torch
from models.model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 设置环境变量以避免OpenMP重复初始化
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    # 设定设备：优先使用 GPU，若无则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 PyTorch Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 获取当前脚本的绝对路径并提取目录部分
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载图像
    img_paths = {
        "sunflower": os.path.join(current_dir, "flower4.jpg"),  # 验证太阳花
        "rose": os.path.join(current_dir, "flower3.jpg")  # 验证玫瑰花
    }

    for name, path in img_paths.items():
        if not os.path.exists(path):
            print(f"Error: 图像文件不存在: {path}")
            continue

        print(f"Loading and processing image: {path}")
        img = Image.open(path).convert('RGB')  # 确保图像为三通道的 RGB 图像
        plt.imshow(img)
        plt.title(f"Original Image - {name.capitalize()}")  # 显示原始图像
        plt.show()

        # 应用数据转换
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)  # 增加批次维度并移动到指定设备

        # 读取类别字典
        class_indices_path = os.path.join(current_dir, 'class_indices.json')
        try:
            with open(class_indices_path, 'r') as f:
                class_indict = json.load(f)
        except FileNotFoundError:
            print(f"Error: 类别索引文件未找到: {class_indices_path}")
            return
        except json.JSONDecodeError:
            print("Error: 类别索引文件损坏，无法解析.")
            return

        # 创建模型实例并加载权重
        model = AlexNet(num_classes=5).to(device)  # 确保模型在正确的设备上
        model_weight_path = os.path.join(os.path.dirname(current_dir), 'models', 'AlexNet.pth')
        try:
            model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 确保权重在正确的设备上
            print("Model weights loaded successfully.")
        except FileNotFoundError:
            print(f"Error: 模型权重文件未找到: {model_weight_path}")
            return
        except RuntimeError:
            print("Error: 模型权重文件与模型结构不匹配.")
            return

        model.eval()  # 设置模型为评估模式

        with torch.no_grad():  # 禁用梯度计算
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).cpu().numpy()  # 将张量移动到 CPU 并转换为 NumPy 数组

        predicted_class = class_indict[str(predict_cla)]
        predicted_probability = predict[predict_cla].item()

        print(f"Predicted class: {predicted_class}, Probability: {predicted_probability:.4f}")

        # 显示预测结果
        img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()  # 将张量转换为 numpy 数组并移回 CPU
        img_np = (img_np * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])) * 255  # 反归一化
        img_np = img_np.astype(np.uint8)
        plt.imshow(img_np)
        plt.title(f"Predicted Class: {predicted_class} ({predicted_probability:.2%})")
        plt.show()


if __name__ == '__main__':
    main()