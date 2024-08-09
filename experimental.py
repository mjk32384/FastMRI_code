import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import torch
import os, sys

import pathlib
import platform
if platform.system() == 'Linux':
  pathlib.WindowsPath = pathlib.PosixPath

if os.getcwd() + '/utils/model/' not in sys.path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, 'utils/model'))

from utils.data.load_data import create_data_loaders
from utils.learning.train_part import validate_acc
from utils.model.varnet import VarNet

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../../home/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../../home/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=6, help='Number of cascades | Should be less than 12') ## important hyperparameter, 1 in original
    parser.add_argument('--chans', type=int, default=12, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter, 9 in original
    parser.add_argument('--sens_chans', type=int, default=5, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter, 4 in original
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

args = parse()

args.exp_dir = '../result' / args.net_name / 'checkpoints'
args.val_dir = '../result' / args.net_name / 'reconstructions_val'
args.main_dir = '../result' / args.net_name / __file__
args.val_loss_dir = '../result' / args.net_name

device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Current cuda device ', torch.cuda.current_device())

model = VarNet(num_cascades=args.cascade, 
                chans=args.chans, 
                sens_chans=args.sens_chans)
model.to(device=device)


checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
model.load_state_dict(checkpoint['model'])

val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
acc_list = [4, 5, 8]
loss_list, time_list = validate_acc(model, val_loader, acc_list)

print(loss_list)
print()
print(time_list)