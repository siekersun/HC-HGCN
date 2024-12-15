import random
import os
import torch
from data.dataset_loader import dataset_loader
from data.pretrain.pretrain_loader import pretrain_loader
from torch_geometric.loader import DataLoader
from model.LCFN_model import LCFN
from model.pretrain_model import LungPrediction
from args import args
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from pre_train import evaluate as eva1
from LFCN_main import evaluate as eva2


if __name__ == "__main__":
    random.seed(args.seed)
    db_train = dataset_loader(base_dir=args.root_path, dataset_class='Training', num_classes=args.num_classes)
    db_test = dataset_loader(base_dir=args.root_path, dataset_class='Internal test', num_classes=args.num_classes)
    Exter_test = dataset_loader(base_dir=args.root_path, dataset_class='External test', num_classes=args.num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    Exterloader = DataLoader(Exter_test, batch_size=1, shuffle=False, num_workers=1)

    # print('resnet+临床结果--------------------------------------------------------------------------------')
    print('临床结果--------------------------------------------------------------------------------')
    net = LungPrediction(input_size=9, hidden_size=args.hidden_size, phi=args.phi,
                             layer_num=args.layer_num, pre_train=args.pre_train, num_class=2).cuda()
    output_dir = os.path.join(args.output_dir, args.output_name)
    state_dict = torch.load(f'{output_dir}/Internal test_model.pt', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    Inter_acc, Inter_loss, p, r, f, Inter_spec, Intest_auroc, Intest_auprc = eva1(net, testloader)
    print(
        'Inter_acc : %f Inter_loss : %f Inter_precise : %f Inter_recall : %f Inter_f1 : %f Inter_spec: %f Inter_auroc : %f Inter_auprc : %f' % (
            Inter_acc, Inter_loss, p, r, f, Inter_spec, Intest_auroc, Intest_auprc))

    state_dict = torch.load(f'{output_dir}/External test_model.pt', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    # Tarin_acc, Tarin_loss, Tarinp, Tarinr, Tarinf, Tarin_auroc, Tarin_auprc = eva1(net, trainloader)
    # print(
    #     'Tarin_acc : %f Tarin_loss : %f Tarin_precise : %f Tarin_recall : %f Tarin_f1 : %f Tarin_auroc : %f Tarin_auprc : %f' % (
    #         Tarin_acc, Tarin_loss, Tarinp, Tarinr, Tarinf, Tarin_auroc, Tarin_auprc))

    Exter_acc, Exter_loss, Exter_p, Exter_r, Exter_f, Exter_spec, Exter_auroc, Exter_auprc = eva1(net, Exterloader)
    print(
        'Exter_acc : %f Exter_loss : %f Exter_precise : %f Exter_recall : %f Exter_f1 : %f Exter_spec: %f Exter_auroc : %f Exter_auprc : %f' % (
            Exter_acc, Exter_loss, Exter_p, Exter_r, Exter_f, Exter_spec, Exter_auroc, Exter_auprc))

    print('再加上超图--------------------------------------------------------------------------------')
    db_train = pretrain_loader(base_dir=args.pretrain_fold, dataset_class='Training', num_classes=args.num_classes)
    inter_test = pretrain_loader(base_dir=args.pretrain_fold, dataset_class='Internal test',
                                num_classes=args.num_classes)
    exter_test = pretrain_loader(base_dir=args.pretrain_fold, dataset_class='External test',
                                 num_classes=args.num_classes)

    trainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=False)
    testloader = DataLoader(inter_test, batch_size=len(inter_test), shuffle=False, num_workers=1)
    Exterloader = DataLoader(exter_test, batch_size=len(exter_test), shuffle=False, num_workers=1)

    net = LCFN(hidden_size=args.hidden_size, phi=args.phi, layer_num=args.layer_num, num_class=2).cuda()

    output_dir = os.path.join(args.output_dir, args.output_name)
    state_dict = torch.load(f'{output_dir}/Internal test_lcfnmodel.pt', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    Inter_acc, Inter_loss, p, r, f, Inter_spec, Intest_auroc, Intest_auprc = eva2(net, testloader)
    print(
        'Inter_acc : %f Inter_loss : %f Inter_precise : %f Inter_recall : %f Inter_f1 : %f Inter_spec: %f Inter_auroc : %f Inter_auprc : %f' % (
            Inter_acc, Inter_loss, p, r, f, Inter_spec, Intest_auroc, Intest_auprc))

    state_dict = torch.load(f'{output_dir}/External test_lcfnmodel.pt', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    # Tarin_acc, Tarin_loss, Tarinp, Tarinr, Tarinf, Tarin_auroc, Tarin_auprc = eva2(net, trainloader)
    # print(
    #     'Tarin_acc : %f Tarin_loss : %f Tarin_precise : %f Tarin_recall : %f Tarin_f1 : %f Tarin_auroc : %f Tarin_auprc : %f' % (
    #         Tarin_acc, Tarin_loss, Tarinp, Tarinr, Tarinf, Tarin_auroc, Tarin_auprc))

    Exter_acc, Exter_loss, Exter_p, Exter_r, Exter_f, Exter_spec, Exter_auroc, Exter_auprc = eva2(net, Exterloader)
    print(
        'Exter_acc : %f Exter_loss : %f Exter_precise : %f Exter_recall : %f Exter_f1 : %f Exter_spec: %f Exter_auroc : %f Exter_auprc : %f' % (
            Exter_acc, Exter_loss, Exter_p, Exter_r, Exter_f, Exter_spec, Exter_auroc, Exter_auprc))