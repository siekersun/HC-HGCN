import torch
import torch.nn as nn
from torchvision import models
from model.LCFN_model import LCFN
from model.Resnet50 import resnet50_3d_cbam
from resnet import generate_model
import torchvision.models.resnet



class LungPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, phi, layer_num, pre_train=True,  dropout=0.4):
        super().__init__()

        self.cnn_cbam = resnet50_3d_cbam(num_classes=2)
        # self.cnn_cbam = generate_model(50,n_classes = 2)
        in_features = self.cnn_cbam.fc.in_features
        self.cnn_cbam.fc = nn.Identity()

        self.num_class = num_class
        self.pre_train = pre_train
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),nn.LeakyReLU(),nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(in_features + hidden_size, 2 * hidden_size),nn.LeakyReLU(),nn.Dropout(dropout))


        # 使用占位符，实际在 forward 中会动态调整输入大小
        self.fc = nn.Linear(2 * hidden_size, num_class)

        # self.layernorm = nn.LayerNorm(hidden_size)


    def forward(self, image, clinical, p):

        x_clinical = self.fc1(clinical)

        x_image = self.cnn_cbam(image)

        x_all = torch.cat([x_image, x_clinical],-1)
        # x_all=x_image
        x_all = self.fc2(x_all)
        
        x = self.fc(x_all)

        if self.num_class > 1:
            return x, x_all
        else:
            return x.flatten(), x_all




