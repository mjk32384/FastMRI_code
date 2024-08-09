import torch
import argparse
import shutil
import os, sys
from pathlib import Path
import h5py
import numpy, cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.data.load_data import create_data_loaders

if __name__=="__main__":
    """
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    train_loader_iter = iter(create_data_loaders(data_path = Path("../../../home/Data/train/"), args = args, shuffle=True))
    first_data = next(train_loader_iter)
    for i, d in enumerate(first_data):
        print(i, end = "***\n")
        print(d)
    """
    fname = "../../../home/Data/train/image/brain_acc4_3.h5"
    with h5py.File(fname, "r") as hf:
        for x in range(16):
            b = hf["image_input"][x] * (10 ** 6)
            cv2.imwrite("image/"+"%03d"%x+ ".jpg", b)
        