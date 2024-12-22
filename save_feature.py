import random
import os
import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.data import DataLoader

from model.pretrain_model import LungPrediction
from args import args
from data.dataset_loader import dataset_loader

if __name__ == '__main__':
    db_train = dataset_loader(base_dir=args.root_path, dataset_class='Training', num_classes=args.num_classes)
    db_test = dataset_loader(base_dir=args.root_path, dataset_class='Internal test', num_classes=args.num_classes)
    Exter_test = dataset_loader(base_dir=args.root_path, dataset_class='External test', num_classes=args.num_classes)
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))
    print("The length of test set is: {}".format(len(Exter_test)))


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    Exterloader = DataLoader(Exter_test, batch_size=1, shuffle=False, num_workers=1)
    # 初始化用于存储所有编码特征的列表
    all_features = []
    output_dir = os.path.join(args.output_dir, args.output_name)
    # 保存测试集的预训练编码
    net = LungPrediction(input_size=9, hidden_size=args.hidden_size, phi=args.phi,
                         layer_num=args.layer_num, pre_train=args.pre_train, num_class=2).cuda()
    state_dict = torch.load(f'{output_dir}/External test_model.pt')
    net.load_state_dict(state_dict)
    net.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(trainloader):
            image, clinical, p, y = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            # 提取编码结果
            _, x_all = net(image, clinical,p)

            # 将当前批次的编码特征添加到列表中
            all_features.append(x_all.cpu().numpy())

    # 将列表转换为单个 NumPy 数组
    all_features = np.concatenate(all_features, axis=0)

    # 转换为 Pandas DataFrame
    df = pd.DataFrame(all_features)

    # 保存到 CSV 文件
    df.to_csv(os.path.join(args.pretrain_fold, "training_features.csv"), index=False)

    print("Encoded features saved successfully to 'training_features.csv'.")

    # 初始化用于存储所有编码特征的列表
    all_features = []

    with torch.no_grad():
        for batch_idx, data in enumerate(Exterloader):
            image, clinical, p, y = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            # 提取编码结果
            _, x_all = net(image, clinical,p)

            # 将当前批次的编码特征添加到列表中
            all_features.append(x_all.cpu().numpy())

    # 将列表转换为单个 NumPy 数组
    all_features = np.concatenate(all_features, axis=0)

    # 转换为 Pandas DataFrame
    df = pd.DataFrame(all_features)

    # 保存到 CSV 文件
    df.to_csv(os.path.join(args.pretrain_fold, "External test_features.csv"), index=False)

    print("Encoded features saved successfully to 'External test_features.csv'.")

    # 初始化用于存储所有编码特征的列表
    all_features = []
    # 保存测试集的预训练编码
    net = LungPrediction(input_size=9, hidden_size=args.hidden_size, phi=args.phi,
                         layer_num=args.layer_num, pre_train=args.pre_train, num_class=2).cuda()
    state_dict = torch.load(f'{output_dir}/Internal test_model.pt')
    net.load_state_dict(state_dict)
    net.eval()
    # 保存测试集的预训练编码
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            image, clinical, p, y = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()

            # 提取编码结果
            _, x_all = net(image, clinical, p)

            # 将当前批次的编码特征添加到列表中
            all_features.append(x_all.cpu().numpy())

    # 将列表转换为单个 NumPy 数组
    all_features = np.concatenate(all_features, axis=0)

    # 转换为 Pandas DataFrame
    df = pd.DataFrame(all_features)

    # 保存到 CSV 文件
    df.to_csv(os.path.join(args.pretrain_fold, "Internal test_features.csv"), index=False)

    print("Encoded features saved successfully to 'Internal test_features.csv'.")
