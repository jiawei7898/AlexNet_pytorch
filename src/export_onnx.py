import torch
from models.model import AlexNet  # 假设你的模型定义在 models/model.py 文件中
import os


def main():
    # 获取当前脚本的绝对路径，并提取出目录部分
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {current_dir}")

    # 模型参数保存路径
    model_path = os.path.join(current_dir, '..', 'models', 'AlexNet.pth')
    print(f"模型路径: {model_path}")

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径 {model_path} 不存在，请确保模型已正确保存。")

    # 创建模型实例
    model = AlexNet(num_classes=5, init_weights=True)
    print("模型实例创建成功。")

    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    print(f"模型权重已从 {model_path} 加载。")

    # 设置模型为评估模式
    model.eval()
    print("模型已设置为评估模式。")

    # 创建一个随机输入张量，假设输入图像大小为 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    print("随机输入张量创建成功。")

    # 指定输出 ONNX 文件名
    output_onnx = os.path.join(current_dir, '..', 'models', 'alexnet.onnx')
    print(f"ONNX 输出路径: {output_onnx}")

    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        export_params=True,  # 存储训练好的参数
        opset_version=11,  # ONNX 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                      'output': {0: 'batch_size'}}
    )
    print(f"模型已成功转换为 ONNX 格式，并保存到 {output_onnx}")


if __name__ == '__main__':
    main()
