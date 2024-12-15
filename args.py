import argparse





parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/cropped_data/', help='root dir for data')
parser.add_argument('--pretrain_fold',type=str,
                    default='data/pretrain')
parser.add_argument('--num_classes', type=int,
                    default=2, help='class number')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size')
parser.add_argument('--layer_num', type=int,
                    default=1)
parser.add_argument('--phi',type=int,
                    default=100, help='Convolutional kernel length')
parser.add_argument('--max_epoch', type=int,
                    default=128, help='max_epoch')
parser.add_argument('--base_lr', type=float,
                    default=1e-2, help='learning rate')
parser.add_argument('--seed', type=int,
                    default=666, help='seed')
parser.add_argument('--hidden_size', type=int,
                    default=128)
parser.add_argument('--output_dir', type=str,
                    default='output', help='output_dir')
parser.add_argument('--output_name', type=str,
                    default='resnet+cbam+clinical', help='class number')
parser.add_argument('--pre_train', type=bool,
                    default=True)

args = parser.parse_args()