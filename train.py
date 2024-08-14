import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='temp', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../../home/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../../home/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=3, help='Number of cascades | Should be less than 12') ## important hyperparameter, 1 in original
    parser.add_argument('--chans', type=int, default=3, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter, 9 in original
    parser.add_argument('--sens_chans', type=int, default=3, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter, 4 in original
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--acc_weight', type=dict, default={4: 1/3, 5: 1/3, 9: 1/3}, help='Probability of each mask')
    # added from here
    parser.add_argument('--previous_model', type=str, default='', help='Previous model(less epoch model) name')
    parser.add_argument('--mask_mode', type=str, default='equispaced', help='Mode of mask applied to data')
    parser.add_argument('--use_SSIM_mask_train', type=bool, default=True, help='use SSIM mask when training')
    # parser.add_argument('--use_SSIM_mask_val', type=bool, default=True, help='use SSIM mask when valdidating')
    # to here
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
