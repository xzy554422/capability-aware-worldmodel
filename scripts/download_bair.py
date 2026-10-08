"""
直接下载 BAIR Robot Pushing 数据集

由于 TFDS 下载速度极慢（~2.6s/MiB），使用直接下载链接。
"""

import os
import sys
import requests
from tqdm import tqdm
import tarfile
import zipfile


def download_file(url: str, dest_path: str, chunk_size: int = 8192):
    """下载文件并显示进度条"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)
    
    print(f"✓ Downloaded to {dest_path}")


def extract_archive(archive_path: str, extract_to: str):
    """解压归档文件"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print(f"✓ Extracted to {extract_to}")


def main():
    """
    BAIR 数据集下载
    
    官方链接（Google Cloud）：
    - 训练集: https://storage.googleapis.com/brain-robotics-data/push/push_train.tar
    - 测试集: https://storage.googleapis.com/brain-robotics-data/push/push_test.tar
    
    每个 tar 包含：
    - traj_0_to_255/ ... traj_X/
      - 每个轨迹文件夹包含 30 帧 .png 图片和 actions.json
    """
    
    data_dir = "data/bair"
    os.makedirs(data_dir, exist_ok=True)
    
    # 数据集 URL（Google Cloud Storage）
    train_url = "https://storage.googleapis.com/brain-robotics-data/push/push_train.tar"
    test_url = "https://storage.googleapis.com/brain-robotics-data/push/push_test.tar"
    
    print("=== BAIR Robot Pushing Dataset Download ===\n")
    print("This will download ~4GB of data from Google Cloud Storage.")
    print(f"Destination: {os.path.abspath(data_dir)}\n")
    
    # 下载训练集
    train_tar = os.path.join(data_dir, "push_train.tar")
    if not os.path.exists(train_tar):
        print("Downloading training set...")
        try:
            download_file(train_url, train_tar)
        except Exception as e:
            print(f"✗ Failed to download training set: {e}")
            print("\nAlternative: Manual download")
            print(f"1. Download from: {train_url}")
            print(f"2. Save to: {train_tar}")
            sys.exit(1)
    else:
        print(f"✓ Training set already downloaded: {train_tar}")
    
    # 下载测试集
    test_tar = os.path.join(data_dir, "push_test.tar")
    if not os.path.exists(test_tar):
        print("\nDownloading test set...")
        try:
            download_file(test_url, test_tar)
        except Exception as e:
            print(f"✗ Failed to download test set: {e}")
            print("\nAlternative: Manual download")
            print(f"1. Download from: {test_url}")
            print(f"2. Save to: {test_tar}")
            sys.exit(1)
    else:
        print(f"✓ Test set already downloaded: {test_tar}")
    
    # 解压
    train_dir = os.path.join(data_dir, "push_train")
    if not os.path.exists(train_dir):
        print("\nExtracting training set...")
        extract_archive(train_tar, data_dir)
    else:
        print(f"✓ Training set already extracted: {train_dir}")
    
    test_dir = os.path.join(data_dir, "push_test")
    if not os.path.exists(test_dir):
        print("\nExtracting test set...")
        extract_archive(test_tar, data_dir)
    else:
        print(f"✓ Test set already extracted: {test_dir}")
    
    print("\n=== Download Complete ===")
    print(f"Training set: {train_dir}")
    print(f"Test set: {test_dir}")
    
    # 统计文件数
    if os.path.exists(train_dir):
        train_trajs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        print(f"\nTraining trajectories: {len(train_trajs)}")
    
    if os.path.exists(test_dir):
        test_trajs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        print(f"Test trajectories: {len(test_trajs)}")
    
    print("\n✓ Dataset ready for use!")
    print("\nNext step: Run `python code/verify_data.py` to verify the data loader")


if __name__ == "__main__":
    main()
