import torch
from torch import nn
import torch.nn.functional as F
from itertools import combinations

def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)


class Self_Attention(nn.Module):
    def __init__(self, in_features, qk_dim, v_dim, head_size):
        super(Self_Attention, self).__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.head_size = head_size

        self.att_dim = att_dim = qk_dim // head_size
        self.scale = att_dim ** -0.5
        self.aru = nn.LeakyReLU()

        self.linear_q = nn.Linear(in_features, qk_dim, bias=False)
        self.linear_k = nn.Linear(in_features, qk_dim, bias=False)
        # self.linear_v = nn.Linear(in_features, self.v_dim, bias=False)

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
        weights = torch.zeros((1, num_edges), dtype=torch.float32).cuda()  # 初始化权重矩阵

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

            weights[0, e] = weight  # 填入权重矩阵中


        return weights

    def forward(self, x1, x2, v, adj, edge_attr):
        if adj.is_sparse:
            adj = adj.to_dense()

        edge_attr = self.compute_hyperedge_weights_torch(x1, adj).T

        batch_size_q = x1.size(0)
        batch_size_k = x2.size(0)

        q = x1.view(batch_size_q, self.head_size, -1)
        k = x2.view(batch_size_k, self.head_size, -1)
        v = v.view(batch_size_k, self.head_size, -1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1)

        q = q * self.scale
        x = torch.matmul(q, k)

        zero_vec = -9e15 * torch.ones_like(adj)
        x = torch.where(adj > 0, x, zero_vec)
        x = torch.softmax(x, dim=-1)
        # x = self.att_dropout(x)
        v = edge_attr * v
        x = x.matmul(v)

        x = x.transpose(0, 1).contiguous()
        x = x.view(batch_size_q, -1)

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.qk_dim,
                                       self.v_dim, self.head_size)


class FeaTrans(nn.Module):
    def __init__(self, in_features, out_features, qk_dim, v_dim, head_size,
                 dropout, bias=True, share=False):
        super(FeaTrans, self).__init__(),
        self.in_features = in_features
        self.out_features = out_features
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.share = share
        self.layer_norm = nn.LayerNorm(out_features)
        self.linear_q = nn.Linear(in_features, qk_dim, bias=False)
        self.linear_k = nn.Linear(in_features, qk_dim, bias=False)

        self.SAT = Self_Attention(in_features=in_features, qk_dim=qk_dim, v_dim=v_dim, head_size=head_size)
        self.weight = nn.Parameter(torch.Tensor(out_features, out_features))

        self.linear_v = nn.Linear(in_features, v_dim, bias=False)
        self.ln = nn.LayerNorm(v_dim)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, x1, x2, x3, edge_index, edge_attr):

        # x = self.layer_norm(x)
        x1 = self.linear_q(x1)
        x2 = self.linear_k(x2)
        v = self.linear_v(x3)

        sat_x = self.SAT(x1, x2, v, edge_index, edge_attr)

        out = sat_x
        # out = torch.matmul(x, self.weight)

        # if self.bias is not None:
        #     out += self.bias
        # out = self.ln(out+x)
        # out = self.leaky_relu(out)
        # out = self.dropout(out)

        return out
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        """
        初始化超图注意力层
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param use_bias: 是否使用偏置项
        """
        super(HypergraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习的参数
        self.W = nn.Parameter(torch.randn(in_features, out_features))  # 权重矩阵
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))       # 注意力向量
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None

    def forward(self, X, H, W):
        """
        前向传播
        :param X: 节点特征矩阵，形状 (N, F)
        :param H: 超图邻接矩阵，形状 (N, E)
        :param W: 超边权重对角矩阵，形状 (E, E)
        :return: 更新后的节点特征矩阵，形状 (N, out_features)
        """
        # 线性变换
        X_transformed = torch.matmul(X, self.W)  # (N, out_features)

        # 消息传递 - 超边到节点的传播
        H_t = H.t()                             # 转置为 (E, N)
        hyperedge_features = torch.matmul(H_t, X_transformed)  # (E, out_features)
        hyperedge_features = torch.matmul(W, hyperedge_features)  # 加权超边特征 (E, out_features)
        node_features = torch.matmul(H, hyperedge_features)       # 回到节点特征 (N, out_features)

        # 注意力机制
        num_nodes = X.size(0)
        a_input = torch.cat([
            X_transformed.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
            X_transformed.repeat(num_nodes, 1)
        ], dim=1)  # (N*N, 2 * out_features)
        e = torch.matmul(a_input, self.a).squeeze()  # 注意力系数 (N*N,)

        # 归一化注意力系数
        e = e.view(num_nodes, num_nodes)
        attention = F.softmax(F.relu(e), dim=1)  # (N, N)

        # 节点特征聚合
        updated_features = torch.matmul(attention, node_features)  # (N, out_features)

        # 添加偏置项（如果有）
        if self.bias is not None:
            updated_features += self.bias

        return updated_features







