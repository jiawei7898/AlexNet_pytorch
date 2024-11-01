# spile_data.py
# AlexNet--CNN经典网络模型详解（pytorch实现）
# ResNet——CNN经典网络模型详解(pytorch实现)
# https://blog.csdn.net/weixin_44023658/article/details/105798326

# 导入必要的库
import os
from shutil import copy
import random


# 定义一个创建文件夹的函数
def mkfile(file):
    if not os.path.exists(file):  # 检查文件夹是否存在
        os.makedirs(file)  # 如果不存在，则创建文件夹
        print(f"创建文件夹: {file}")
    else:
        print(f"文件夹 {file} 已经存在。")


# 获取当前脚本的绝对路径，并提取出目录部分
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集路径
data_dir = os.path.join(current_dir, '..', 'data')
flower_photos_dir = os.path.join(data_dir, 'flower_photos')

# 获取所有类别名称，排除.txt文件
flower_class = [cla for cla in os.listdir(flower_photos_dir) if ".txt" not in cla]

# 创建训练集文件夹
train_dir = os.path.join(data_dir, 'train')
mkfile(train_dir)
for cla in flower_class:
    class_train_dir = os.path.join(train_dir, cla)
    mkfile(class_train_dir)  # 为每个类别创建子文件夹

# 创建验证集文件夹
val_dir = os.path.join(data_dir, 'val')
mkfile(val_dir)
for cla in flower_class:
    class_val_dir = os.path.join(val_dir, cla)
    mkfile(class_val_dir)  # 为每个类别创建子文件夹

# 定义数据集划分比例
split_rate = 0.1

# 遍历每个类别
for cla in flower_class:
    cla_path = os.path.join(flower_photos_dir, cla)  # 当前类别的路径
    images = os.listdir(cla_path)  # 获取当前类别下的所有图片
    num = len(images)  # 图片总数

    # 随机选择一部分图片作为验证集
    eval_index = random.sample(images, k=int(num * split_rate))

    # 遍历所有图片
    for index, image in enumerate(images):
        image_path = os.path.join(cla_path, image)  # 图片的完整路径

        if image in eval_index:  # 如果图片在验证集中
            new_path = os.path.join(val_dir, cla)  # 新的路径（验证集）
            copy(image_path, new_path)  # 复制图片到新的路径
            print(f"复制图片 {image} 到验证集 {new_path}")
        else:  # 如果图片不在验证集中
            new_path = os.path.join(train_dir, cla)  # 新的路径（训练集）
            copy(image_path, new_path)  # 复制图片到新的路径
            print(f"复制图片 {image} 到训练集 {new_path}")

        # 打印处理进度
        print(f"\r[类别 {cla}] 处理进度: [{index + 1}/{num}]", end="")  # 进度条

    print()  # 每个类别处理完后换行

print("数据集分割完成！")  # 所有处理完成后打印完成信息
