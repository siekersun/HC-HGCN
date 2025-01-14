
import torch
import torch.nn as nn
import torch.nn.init as init
from .HGAT import HypergraphAttentionLayer, FeaTrans
from .HGCN import HGCN
class HGNN(nn.Module):
    def __init__(self,hidden_size=128, phi=50, layer_num=3, dropout=0.3, num_class=2):
        super(HGNN, self).__init__()

        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.phi = phi
        self.num_class = num_class


        self.emb = nn.ModuleList()
        self.filters = nn.ParameterList()

        for _ in range(layer_num):
            embedding = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),nn.LayerNorm(self.hidden_size), nn.LeakyReLU())
            self.emb.append(embedding)

        for _ in range(self.layer_num):
            # 创建一个对角矩阵的对角元素，并将其转换为对角矩阵
            filter_diag = torch.randn(self.phi) * 0.001 + 1.0
            filter = nn.Parameter(torch.diag(filter_diag))
            self.filters.append(filter)

        # self.trans = nn.ModuleList()
        # for _ in range(self.layer_num):
        #     self.trans.append(FeaTrans(in_features=hidden_size, out_features=hidden_size, qk_dim=hidden_size,
        #                                 v_dim=hidden_size, head_size=2, dropout=dropout))

        self.hgcns = nn.ModuleList()
        for _ in range(self.layer_num):
            self.hgcns.append(HGCN(in_features=hidden_size, out_features=hidden_size))

        self.fc1 = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size),  nn.Dropout(dropout))
        # self.fc2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size),  nn.Dropout(dropout))

        # self.fc3 = nn.Sequential(nn.Linear(1 * self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size),
        #                          nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(self.hidden_size, self.hidden_size))
        self.fc3 = nn.Sequential(nn.Linear((self.layer_num+1) * self.hidden_size, self.hidden_size), nn.LayerNorm(self.hidden_size),
                                 nn.LeakyReLU())

        self.fc = nn.Linear(self.hidden_size, self.num_class)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    def forward(self, x_all, edge_index, p):


        x = self.fc1(x_all)
        # x = self.fc2(x)
        h = x
        for i in range(self.layer_num):

            ##LCFN
            # h = p @ self.filters[i] @ p.T @ h

            #HGCN
            h = self.hgcns[i](x,edge_index)

            h = self.emb[i](h)
            x = torch.cat([x,h],-1)
            # x = (x + h)/2
        x = self.fc3(x)
        x = self.fc(x)

        return x









