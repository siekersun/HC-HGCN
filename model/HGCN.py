import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from math import sqrt

def norm(hypergraph, W):
    # H = torch.tensor(hypergraph, dtype=torch.float32).clone().detach()
    H = hypergraph
    epsilon = 0.1 ** 10
    tolerant = 0.1 ** 5

    num_nodes = H.shape[0]
    num_edges = H.shape[1]
    Du = torch.sum(H, axis=1).reshape(-1)  # Degree of nodes
    Di = torch.sum(H, axis=0).reshape(-1)  # Degree of edges

    I = torch.eye(num_nodes, dtype=torch.float32).cuda()

    Dn = torch.zeros((num_nodes, num_nodes), dtype=torch.float32).cuda()
    De = torch.zeros((num_edges, num_edges), dtype=torch.float32).cuda()


    # Compute Dn, De, and W
    for u in range(num_nodes):
        Dn[u, u] = 1.0 / max(sqrt(Du[u].item()), epsilon)

    for i in range(num_edges):
        De[i, i] = 1.0 / max(Di[i].item(), epsilon)

    # W is a predefined diagonal matrix, can be set as a torch tensor
    W = torch.diag(W)

    # Compute the Laplacian matrix L
    norm_edge = torch.matmul(torch.matmul(torch.matmul(Dn, H), W), torch.matmul(De, torch.matmul(H.T, Dn)))


    return norm_edge

class HGCN(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

    def compute_hyperedge_weights_torch(self, X, H):
        """
        使用 PyTorch 计算超边权重 w(e)，返回一个矩阵

        参数:
        H: torch.Tensor, 超图关联矩阵，形状为 (n, m)，n 为节点数，m 为超边数
        X: torch.Tensor, 节点特征矩阵，形状为 (n, d)，d 为节点特征维度

        返回:
        weights: torch.Tensor, 形状为 (1, m)，每个超边对应的权重 w(e)
        """
        num_edges = H.shape[1]  # 超边数
        weights = torch.zeros((num_edges), dtype=torch.float32).cuda()  # 初始化权重矩阵

        for e in range(num_edges):
            # 获取当前超边中的节点索引
            node_indices = torch.where(H[:, e] > 0)[0]
            num_nodes = len(node_indices)

            if num_nodes < 2:  # 如果超边中的节点数小于2，权重设置为0
                continue

            # 提取超边中所有节点的特征
            node_features = X[node_indices]

            # 计算节点对之间的欧式距离平方
            distances = []
            for u, v in combinations(range(num_nodes), 2):
                dist = torch.norm(node_features[u] - node_features[v], p=2) ** 2
                distances.append(dist)

            distances = torch.stack(distances)  # 转为 PyTorch 张量

            # 计算 σ（距离的中位数）
            sigma = torch.median(distances)

            # 计算权重 w(e)
            weight = (1 / (num_nodes * (num_nodes - 1))) * torch.sum(torch.exp(-distances / (sigma ** 2)))

            assert not torch.isnan(weight).any(), "Tensor contains NaN values!"

            weights[e] = weight  # 填入权重矩阵中


        return weights

    def forward(self, x, edge_index):

        if edge_index.is_sparse:
            adj = edge_index.to_dense()

        weights = self.compute_hyperedge_weights_torch(x, edge_index)

        norm_edge = norm(edge_index, weights)

        x = norm_edge @ x

        return x
