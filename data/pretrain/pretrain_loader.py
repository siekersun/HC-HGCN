from torch.utils.data import Dataset
import os
import pandas as pd
import torch

class pretrain_loader(Dataset):
    def __init__(self, base_dir, dataset_class, num_classes):
        """
        Args:
            base_dir (str): 数据集的基本路径
            dataset_class (str): 数据集的类别 (如 'train' 或 'test')
            num_classes (int): 类别数
        """
        self.num_classes = num_classes
        self.dataset_class = dataset_class
        self.base_dir = base_dir
        self.root_path = f'data/cropped_data/{self.dataset_class} set'

        # 路径设置
        self.pre_feature_path = os.path.join(self.base_dir, f'{self.dataset_class}_features.csv')
        self.hypergraph = os.path.join(self.base_dir, f'{self.dataset_class}_hypergraph_embeddings.csv')
        self.label_path = os.path.join(self.base_dir, f'{self.dataset_class}_label.csv')
        self.clinical_data = pd.read_csv(os.path.join(self.base_dir, f"{self.dataset_class}.csv")).iloc[:, 2:].to_numpy()

        # 读取数据并缓存为类属性
        self.features = pd.read_csv(self.pre_feature_path).to_numpy()
        self.p = pd.read_csv(self.hypergraph).to_numpy()
        self.label = pd.read_csv(self.label_path)['result'].to_numpy()

        # 数据一致性检查
        assert len(self.features) == len(self.label), "数据错误：特征数据和标签数量不一致！"

        # 将数据转换为张量
        self.features = torch.tensor(self.features).float()
        self.p = torch.tensor(self.p).float()
        self.label = torch.tensor(self.label).long()
        self.clinical_data = torch.tensor(self.clinical_data).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.clinical_data[idx], self.p[idx], self.label[idx]
