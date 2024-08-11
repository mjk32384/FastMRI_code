import torch
import argparse
import shutil
import os, sys
from pathlib import Path
import json
#from ast import literal_eval

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse(params):
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=params['epoch'], help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=params['learning_rate'], help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=params['report_interval'], help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default=params['net_name'], help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../../home/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../../home/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=params['cascade'], help='Number of cascades | Should be less than 12') ## important hyperparameter, 1 in original
    parser.add_argument('--chans', type=int, default=params['chans'], help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter, 9 in original
    parser.add_argument('--sens_chans', type=int, default=params['sens_chans'], help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter, 4 in original
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--acc_weight', type=dict, default=params['acc_weight'], help='Probability of each mask')
    parser.add_argument('--previous_model', type=str, default=params['previous_model'], help='Previous model(less epoch model) name')
    parser.add_argument('--previous_model_epoch', type=str, default=params['previous_model_epoch'], help='Previous model(less epoch model) name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('params.json', 'r') as f:
        params = json.load(f)
    params['acc_weight'] = {int(k):v for k,v in params['acc_weight'].items()} #json saves dict keys to str, change key names to integer
    print("Training Start")
    print(params)

    args = parse(params)
    
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
