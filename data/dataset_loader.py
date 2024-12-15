import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom


def load_all_dicom_slices(folder_path):
    """从指定文件夹中读取所有DICOM文件，并按顺序堆叠为3D体数据"""
    slices = []
    dcm_files = [f for f in os.listdir(folder_path) if f.endswith('.DCM')]

    # 确保按照文件名顺序读取（可以根据需求修改排序逻辑）
    dcm_files = sorted(dcm_files)

    for filename in dcm_files:
        dcm_path = os.path.join(folder_path, filename)
        dcm = pydicom.dcmread(dcm_path)
        img_array = dcm.pixel_array
        img_array = np.expand_dims(img_array, axis=0)  # 添加通道维度
        slices.append(img_array)

    # 将2D切片堆叠为3D体数据 (depth, height, width)
    volume = np.array(slices)
    # print(f"Loaded volume shape: {volume.shape}")

    return volume

class dataset_loader(Dataset):
    def __init__(self, base_dir, dataset_class, num_classes):
        """
        Args:
            base_dir (str): 数据集的基本路径
            dataset_class (str): 数据集的类别 (如 'train' 或 'test')
            num_classes (int): 类别数
        """
        self.num_classes = num_classes
        self.dataset_class = dataset_class
        self.base_dir = os.path.join(base_dir, f'{dataset_class} set')
        self.hypergraph_dir = 'data/pretrain'

        # 读取标签文件
        self.hypergraph_embedding = pd.read_csv(os.path.join(self.hypergraph_dir, f"{self.dataset_class}_hypergraph_embeddings.csv")).to_numpy()
        self.data_df = pd.read_csv(os.path.join(self.base_dir, "label.csv"),dtype={'Patient ID': str})
        self.clinical_data = pd.read_csv(os.path.join(self.base_dir, f"{self.dataset_class}.csv")).iloc[:, 2:].to_numpy()

        # 数据一致性检查
        assert len(self.data_df) == len(self.clinical_data), "数据错误：特征数据和标签数量不一致！"

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # 获取患者的 ID、结节信息和标签
        data_id = self.data_df['Patient ID'].iloc[idx]
        data_nodule = self.data_df['Nodule ID'].fillna('label1').iloc[idx]
        data_label = self.data_df['result'].iloc[idx]
        # 构建患者图像数据的路径
        patient_path = os.path.join(self.base_dir, f"{data_id}/{data_nodule}.npy")

        # 加载 3D 图像数据
        image_data = np.load(patient_path)
        image_data = np.expand_dims(image_data, axis=0)  # 添加通道维度，使形状变为 (C, D, H, W)

        # 获取临床特征
        clinical_feature = self.clinical_data[idx]
        p = torch.tensor(self.hypergraph_embedding)[idx].float()

        # 将数据转换为 PyTorch tensors
        image_tensor = torch.tensor(image_data, dtype=torch.float32)
        clinical_tensor = torch.tensor(clinical_feature, dtype=torch.float32)
        label_tensor = torch.tensor(data_label, dtype=torch.long)
        return image_tensor, clinical_tensor, p, label_tensor


