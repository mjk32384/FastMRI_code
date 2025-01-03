import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
#added here
import tqdm
# added here
import cv2

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

# added from here
class EpochTracker:
    """
    현재 epoch을 전달해주는 class
    epoch에 따라서 Data Augmentation을 다르게 하기 위해서 만듦
    """
    def __init__(self):
        self.epoch = 0

    def get_epoch(self):
        return self.epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

def create_ssim_mask(target):
    """
    SSIM mask 만드는 함수.
    leaderboard_eval에 있는 코드 가져옴
    """
    ssim_mask = np.zeros(target.shape[1:])
    ssim_mask[target.cpu()[0]>5e-5] = 1
    kernel = np.ones((3, 3), np.uint8)
    ssim_mask = cv2.erode(ssim_mask, kernel, iterations=1)
    ssim_mask = cv2.dilate(ssim_mask, kernel, iterations=15)
    ssim_mask = cv2.erode(ssim_mask, kernel, iterations=14)
    return ssim_mask
# to here

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask) # shape: [1, 384, 384]

        # added from here
        if args.use_SSIM_mask_train:
            ssim_mask = create_ssim_mask(target)
            ssim_mask = torch.tensor(ssim_mask)
            ssim_mask = ssim_mask.cuda(non_blocking=True)
            target = (target * ssim_mask).type(torch.float)
            output = (output * ssim_mask).type(torch.float)

        # loss가 SSIM이면 위, L1 or MSE면 아래
        try:
            loss = loss_type(output, target, maximum)
        except:
            loss = loss_type(output, target)
        # to here
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    # added this line
    ssim_masks = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask) # shape: [1, 384, 384]
            # added this line
            ssim_mask = create_ssim_mask(target)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()\
                # added this line
                ssim_masks[fnames[i]][int(slices[i])] = ssim_mask

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    # added from here
    for fname in ssim_masks:
        ssim_masks[fname] = np.stack(
            [out for _, out in sorted(ssim_masks[fname].items())]
        )
    # to here
    
    # 이 ssim_loss는 slice들의 평균이 아닌, h5파일들의 평균. leaderboard_eval에서는 slice들의 평균을 계산함
    # utils.common.loss_function의 SSIMLoss와 leaderboard_eval의 SSIM과 SSIM 구하는 방법은 같음
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])

    #added this line
    metric_loss_mask = sum([ssim_loss(targets[fname]*ssim_masks[fname], reconstructions[fname]*ssim_masks[fname]) for fname in reconstructions])
    
    num_subjects = len(reconstructions)
    return metric_loss, metric_loss_mask, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)
    

    """
    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
    
    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]
    model.load_state_dict(pretrained)
    """
    # added from here
    if args.loss_type == 'SSIM':
        loss_type = SSIMLoss().to(device=device)
    elif args.loss_type == 'MSE':
        loss_type = nn.MSELoss().to(device=device)
    elif args.loss_type == 'L1':
        loss_type = nn.L1Loss().to(device=device)
    else:
        raise Exception("Invalid loss type")
    # to here

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0
    
    # added this line
    epoch_tracker = EpochTracker()
    # edited this line
    train_loader = create_data_loaders(data_path = args.data_path_train,
                                       args = args,
                                       shuffle=True,
                                       augment = args.augment,
                                       epoch_fn = epoch_tracker.get_epoch,
                                       add_gaussian_noise=args.add_gaussian_noise)
    
    acc_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # edit: val_loss_log -> val_loss_log.npy
    file_path = os.path.join(args.val_loss_dir, "val_loss_log.npy")
    # added this line
    file_path_mask = os.path.join(args.val_loss_dir, "val_loss_mask_log.npy")
    try:
        val_loss_log = np.load(file_path)
    except:
        val_loss_log = np.empty((0, len(acc_list)+1))
    # added from here
    try:
        val_loss_mask_log = np.load(file_path_mask)
    except:
        val_loss_mask_log = np.empty((0, len(acc_list)+1))
    # to here
    
    file_path_lr = os.path.join(args.val_loss_dir, "lr_log.npy")
    lr_list = np.empty(shape = (0, 2))
    
    if(args.previous_model):
        file_path_lr_prev = '/'.join(str(args.val_loss_dir).split('/')[:-1]) + '/' + args.previous_model + '/lr_log.npy'
        try:
            lr_list = np.load(file_path_lr_prev)
            args.lr = lr_list[-1][1]
        except:
            print('previous model learning rate load failed.. uh-oh')
    
    # created here
    try:
        if(args.previous_model): #import previous model
            print('/'.join(str(args.val_loss_dir).split('/')[:-1]) + '/' + args.previous_model + '/model.pt')
            checkpoint = torch.load('/'.join(str(args.val_loss_dir).split('/')[:-1]) + '/' + args.previous_model + '/checkpoints/model.pt', map_location='cpu')
            print(checkpoint['epoch'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("Model imported : " + args.previous_model)
    except:
        pass
    # to here

    # created here
    prev_train_loss = np.inf
    loss_increase_epoch = 0
    # to here

    for epoch in range(start_epoch, args.num_epochs):
        #added here
        lr_list = np.append(lr_list, np.array([[epoch, args.lr]]), axis = 0)
        np.save(file_path_lr, lr_list)
        
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        # created here
        print(f'Current learning rate {args.lr}')
        # to here
        
        # added this line
        # 일단 epoch이 커질수록 Data Augmentation이 더 많이되도록 설정했는데, 일단 테스트해보기 위해서 epoch=50으로 설정했음
        # epoch_tracker.set_epoch(epoch+1)
        epoch_tracker.set_epoch(50)

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        
        val_loss_list = []
        val_loss_mask_list = []
        val_time_list = []


        for acc in acc_list:
            val_loader = create_data_loaders(data_path = args.data_path_val, args = args, validate = True, acc = acc)
            val_loss, val_loss_mask, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
            val_loss_list.append(val_loss/num_subjects)
            # added this line
            val_loss_mask_list.append(val_loss_mask/num_subjects)
            val_time_list.append(val_time)
        val_loss_list.insert(0, epoch)
        # added this line
        val_loss_mask_list.insert(0, epoch)
        val_loss_log = np.append(val_loss_log, np.array([val_loss_list]), axis=0)
        # added this line
        val_loss_mask_log = np.append(val_loss_mask_log, np.array([val_loss_mask_list]), axis=0)
        np.save(file_path, val_loss_log)
        # added this line
        np.save(file_path_mask, val_loss_mask_log)
        print(f"loss file saved! {file_path}")

        # edited here
        # ssim mask 적용된 validation loss로 val_loss계산
        val_loss = np.mean(val_loss_mask_list[1:])

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {sum(val_time_list):.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
               f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            
        # added from here
        if prev_train_loss < train_loss:
            loss_increase_epoch += 1
        else:
            loss_increase_epoch = 0
        if loss_increase_epoch == 2:
            args.lr = args.lr / 2
            print("learning rate halved - new learning rate %f"%args.lr)
        prev_train_loss = train_loss
        # to here