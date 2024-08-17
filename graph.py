import numpy as np
import torch
from utils.common.loss_function import SSIMLoss
import torch.nn.functional as F
import os, sys
from pathlib import Path
import argparse, time, json
import h5py, cv2
from collections import defaultdict
import matplotlib.pyplot as plt

if os.getcwd() + '/utils/model/' not in sys.path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, 'utils/model'))

from utils.model.varnet import VarNet
from utils.data.load_data import create_mask, create_data_loaders
from utils.data.transforms import DataTransform
from utils.common.utils import ssim_loss


def parse(arguments):
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_6125_59', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../../home/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../../home/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=6, help='Number of cascades | Should be less than 12') ## important hyperparameter, 1 in original
    parser.add_argument('--chans', type=int, default=12, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter, 9 in original
    parser.add_argument('--sens_chans', type=int, default=5, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter, 4 in original
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--acc_weight', type=json.loads, default={4: 1/3, 5: 1/3, 9: 1/3}, help='Probability of each mask')
    parser.add_argument('--previous_model', type=str, default='', help='Previous model(less epoch model) name')
    parser.add_argument('--mask_mode', type=str, default='equispaced', help='Mode of mask applied to data')

    args = parser.parse_args(arguments)
    return args

# val_loss_log로부터 그래프 그려주는 함수
def graph_val_loss(data, label):
    fig = plt.figure(figsize=(10, 13))
    grid = plt.GridSpec(4, 3, hspace= 0.4, wspace= 0.4)
    ax = []
    for i in range(9):
        if i == 0:
            ax.append(fig.add_subplot(grid[i//3, i%3]))
        else:
            ax.append(fig.add_subplot(grid[i//3, i%3], sharex = ax[0], sharey = ax[0]))
        for j, df in enumerate(data):
            ax[i].plot(range(1, len(df)+1), df[:, i+1], label = label[j])
        ax[i].set_title(f"acc{i+2}")
        # ax[i].legend()
    total_ax = fig.add_subplot(grid[3:,:])
    for j, df in enumerate(data):
        total_ax.plot(range(2, 11), df[-1,1:], label = label[j])
    total_ax.set_xlabel("acc")
    total_ax.legend()

# train_part에 있는 거 변형. slice별로 SSIM 알 수 있게
def validate(model, data_loader, use_SSIM_mask = False):
    model.eval()
    ssim_losses = defaultdict(dict)
    
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)


            if use_SSIM_mask:
                ssim_mask = np.zeros(target.shape[1:])
                ssim_mask[target[0]>5e-5] = 1
                kernel = np.ones((3, 3), np.uint8)
                ssim_mask = cv2.erode(ssim_mask, kernel, iterations=1)
                ssim_mask = cv2.dilate(ssim_mask, kernel, iterations=15)
                ssim_mask = cv2.erode(ssim_mask, kernel, iterations=14)

                target = target * ssim_mask
                output = output.cpu() * ssim_mask
            

            # output.shape[0] = 1
            for i in range(output.shape[0]):
                ssim_losses[fnames[i]][int(slices[i])] = ssim_loss(target[i].unsqueeze(0).numpy(), output[i].cpu().unsqueeze(0).numpy())
    return ssim_losses, time.perf_counter() - start

# leaderboard_eval에 있는 거에서 return만 바꿈
class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        #data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S

# slice 하나의 reconsrtuction을 해주는 함수
def reconstruction_slice(args, device):
    transform = DataTransform(isforward=False, max_key='max')
    dataslice = args.dataslice

    with h5py.File(args.image_fname, "r") as hf:
        target = hf['image_label'][dataslice]
        attrs = dict(hf.attrs)
    with h5py.File(args.kspace_fname, "r") as hf:
        input = hf['kspace'][dataslice]
        mask = create_mask(args.acc_weight, hf["kspace"].shape[-1], args.mask_mode)
        
    mask, kspace, target, maximum, fname, slice = transform(mask, input, target, attrs, args.kspace_fname, dataslice)
    mask = mask.unsqueeze(0)
    kspace = kspace.unsqueeze(0)

    model = VarNet(num_cascades=args.cascade, 
                    chans=args.chans, 
                    sens_chans=args.sens_chans)
    model.to(device=device)

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()

    with torch.no_grad():
        kspace = kspace.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        recon = model(kspace, mask)
    target = target.cuda(non_blocking=True)
    
    return recon, target, maximum


# 픽셀별 SSIM 구해주는 함수
def SSIM_by_pixel(args):
    args.exp_dir = '../result' / args.net_name / 'checkpoints'    

    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    ssim_calculator = SSIM().to(device=device)

    recon, target, maximum = reconstruction_slice(args, device)

    ssim_graph = ssim_calculator(recon[0], target, maximum).cpu().numpy()
    return ssim_graph[0][0]

# 슬라이스별 SSIM 구해주는 함수
def SSIM_by_slice(args):
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    data_path = args.data_path_val
    data_loader = create_data_loaders(data_path, args, shuffle=False, isforward=False, default_acc=False, validate=False, acc=None)

    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                    chans=args.chans, 
                    sens_chans=args.sens_chans)
    model.to(device=device)

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    ssim_losses, val_time = validate(model, data_loader, use_SSIM_mask = args.use_SSIM_mask)
    return ssim_losses, val_time