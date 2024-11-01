import os
import urllib.request
import tarfile

# 定义数据集的下载链接
DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'


def download_data(url, dest_dir):
    """
    下载并解压数据集到指定目录。

    :param url: 数据集下载链接
    :param dest_dir: 目标目录
    """
    try:
        # 获取文件名
        filename = os.path.basename(url)
        # 构建文件的完整路径
        filepath = os.path.join(dest_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"文件 {filename} 已经存在于 {filepath}，跳过下载。")
        else:
            # 开始下载文件
            print(f"开始下载 {filename} 到 {filepath}...")
            urllib.request.urlretrieve(url, filepath, reporthook=download_progress)
            print(f"{filename} 下载完成！")

        # 检查解压后的文件夹是否已经存在
        extracted_folder_path = os.path.join(dest_dir, 'flower_photos')
        if os.path.exists(extracted_folder_path):
            print(f"解压后的文件夹 {extracted_folder_path} 已经存在，跳过解压步骤。")
        else:
            print(f"正在解压文件 {filename} 到 {dest_dir}...")
            # 使用 tarfile 模块解压文件
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=dest_dir)
            print(f"解压完成！解压后的文件位于 {extracted_folder_path}")

    except Exception as e:
        print(f"发生错误: {e}")


def download_progress(count, block_size, total_size):
    """
    显示下载进度的回调函数。

    :param count: 已经传输的数据块数目
    :param block_size: 数据块大小（通常为8192字节）
    :param total_size: 远程文件的总大小
    """
    percent = int(count * block_size * 100 / total_size)
    print(f"\r{'=' * int(percent / 2)}>{'.' * (50 - int(percent / 2))} {percent}% ", end="")
    if percent >= 100:
        print()


if __name__ == '__main__':
    try:
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 设置数据集保存的目录
        data_dir = os.path.join(current_dir, '..', 'data')

        print(f"当前脚本路径: {current_dir}")
        print(f"数据集保存路径: {data_dir}")

        # 确保 data 目录存在
        if not os.path.exists(data_dir):
            print(f"数据集保存目录 {data_dir} 不存在，正在创建...")
            os.makedirs(data_dir)
            print(f"数据集保存目录 {data_dir} 创建成功！")
        else:
            print(f"数据集保存目录 {data_dir} 已经存在。")

        # 调用 `download_data` 函数
        download_data(DATA_URL, data_dir)

    except Exception as e:
        print(f"发生错误: {e}")
