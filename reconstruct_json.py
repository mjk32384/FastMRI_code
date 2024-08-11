import argparse
from pathlib import Path
import os, sys
import json

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward
import time

#for only linux-based systems
import pathlib
import platform
if platform.system() == 'Linux':
  pathlib.WindowsPath = pathlib.PosixPath
    
def parse(params):
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default=params['net_name'], help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='../../home/Data/leaderboard/', help='Directory of test data')
    
    parser.add_argument('--cascade', type=int, default=params['cascade'], help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default= params['chans'], help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=params['sens_chans'], help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with open('params.json', 'r') as f:
        params = json.load(f)
    params['acc_weight'] = {int(k):v for k,v in params['acc_weight'].items()} #json saves dict keys to str, change key names to integer
    print("Reconstruction Start")

    args = parse(params)
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    public_acc, private_acc = None, None

    assert(len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
      if acc in ['acc4', 'acc5', 'acc8']:
        public_acc = acc
      else:
        private_acc = acc
        
    assert(None not in [public_acc, private_acc])
    
    start_time = time.time()
    
    # Public Acceleration
    args.data_path = args.path_data / public_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'public'
    args.acc_weight = {5: 1}
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    # Private Acceleration
    args.data_path = args.path_data / private_acc # / "kspace"    
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / 'private'
    args.acc_weight = {9: 1}
    print(f'Saved into {args.forward_dir}')
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')
    
    print('Success!') if reconstructions_time < 3000 else print('Fail!')