import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from models.model import AlexNet
import os
import json
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设备：GPU 或 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据转换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),  # 必须是 (224, 224)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 获取当前脚本的绝对路径，并提取出目录部分
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据根目录
data_root = os.path.join(current_dir, '..', 'data')
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')

# 检查数据集路径是否存在
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"训练集路径 {train_dir} 不存在，请确保数据集已正确准备。")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"验证集路径 {val_dir} 不存在，请确保数据集已正确准备。")

# 加载训练集
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
train_num = len(train_dataset)
print(f"训练集样本数: {train_num}")

# 加载验证集
validate_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform["val"])
val_num = len(validate_dataset)
print(f"验证集样本数: {val_num}")

# 类别索引
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# 写入 JSON 文件
json_str = json.dumps(cla_dict, indent=4)
class_indices_path = os.path.join(current_dir, 'class_indices.json')
with open(class_indices_path, 'w') as json_file:
    json_file.write(json_str)
print(f"类别索引已保存到 {class_indices_path}")

# 输出类别索引映射
print(f"类别索引映射: {flower_list}")

# 混淆矩阵存放位置
confusion_matrix_dir = os.path.join(current_dir, '..', 'confusion_matrix')
os.makedirs(confusion_matrix_dir, exist_ok=True)
print(f"混淆矩阵存放位置: {confusion_matrix_dir}")

# 数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)  # 在 Windows 上，num_workers 应该设置为 0
print(f"训练数据加载器已创建，批次大小: {batch_size}")
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)  # 在 Windows 上，num_workers 应该设置为 0
print(f"验证数据加载器已创建，批次大小: {batch_size}")

# 创建模型
net = AlexNet(num_classes=5, init_weights=True)
net.to(device)
print("AlexNet模型已创建并移至设备")

# 损失函数：这里用交叉熵
loss_function = nn.CrossEntropyLoss()
print("交叉熵损失函数已定义")

# 优化器：这里用 Adam
optimizer = optim.Adam(net.parameters(), lr=0.0001)
print("Adam优化器已定义")

# 训练参数保存路径
save_path = os.path.join(current_dir, '..', 'models', 'AlexNet.pth')
print(f"模型保存路径: {save_path}")

# 训练过程中最高准确率
best_acc = 0.0

# 总训练时间
total_start_time = time.time()

# 记录每个 epoch 的损失和准确率
train_losses = []
val_accuracies = []
epoch_train_times = []

# 开始进行训练和测试，训练一轮，测试一轮
num_epochs = 100

# 创建本次训练的混淆矩阵文件夹
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
cm_output_dir = os.path.join(confusion_matrix_dir, f'training_{timestamp}')
os.makedirs(cm_output_dir, exist_ok=True)
print(f"混淆矩阵文件夹已创建: {cm_output_dir}")

for epoch in range(num_epochs):
    # 训练
    net.train()  # 训练过程中，使用之前定义网络中的 dropout
    running_loss = 0.0
    t1 = time.time()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        # 打印训练过程
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(f"\rEpoch {epoch + 1}/{num_epochs} - 训练损失: {int(rate * 100):^3.0f}% [{a}->{b}] {loss:.4f}", end="")

    avg_train_loss = running_loss / (step + 1)
    print(f"\rEpoch {epoch + 1}/{num_epochs} - 平均训练损失: {avg_train_loss:.4f}", end="")

    # 单轮训练时间
    epoch_train_time = time.time() - t1
    epoch_train_times.append(epoch_train_time)
    print(f" - 本轮训练时间: {epoch_train_time:.2f}s")

    # 验证
    net.eval()  # 测试过程中不需要 dropout，使用所有的神经元
    acc = 0.0  # 累计准确数量 / epoch
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = net(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels).sum().item()
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch + 1

        # 记录每个 epoch 的损失和准确率
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accurate)

        # 输出训练和验证结果
        print(f'\nEpoch {epoch + 1}/{num_epochs} - 验证精度: {val_accurate:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs} - 训练精度: {acc / train_num:.4f}')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"混淆矩阵:\n{cm}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cla_dict.values(), yticklabels=cla_dict.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Epoch {epoch + 1})')

    # 保存混淆矩阵图
    cm_filename = f'confusion_matrix_epoch_{epoch + 1}.png'
    cm_path = os.path.join(cm_output_dir, cm_filename)
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵图已保存到 {cm_path}")

# 总训练时间
total_train_time = time.time() - total_start_time
print(f'总训练时间: {total_train_time:.2f}s')

# 输出最佳准确率并保存模型
print(f"最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch})")
torch.save(net.state_dict(), save_path)
print(f"最佳模型已保存到 {save_path}")

# 创建新的文件夹来存放运行产生的参数变化图
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
output_dir = os.path.join(current_dir, '..', 'models', f'train_logs_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
print(f"参数变化图文件夹已创建: {output_dir}")

# 绘制训练损失图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training Loss Over Epochs\nTotal Training Time: {total_train_time:.2f}s')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(output_dir, 'training_loss.png')
plt.savefig(loss_plot_path)
plt.close()
print(f"训练损失图已保存到 {loss_plot_path}")

# 绘制验证准确率图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Validation Accuracy Over Epochs\nBest Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})')
plt.legend()
plt.grid(True)
accuracy_plot_path = os.path.join(output_dir, 'validation_accuracy.png')
plt.savefig(accuracy_plot_path)
plt.close()
print(f"验证准确率图已保存到 {accuracy_plot_path}")

# 绘制每轮训练时间图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_train_times, label='Training Time per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Time (s)')
plt.title(f'Training Time per Epoch\nTotal Training Time: {total_train_time:.2f}s')
plt.legend()
plt.grid(True)
time_plot_path = os.path.join(output_dir, 'training_time_per_epoch.png')
plt.savefig(time_plot_path)
plt.close()
print(f"每轮训练时间图已保存到 {time_plot_path}")

print('训练完成')