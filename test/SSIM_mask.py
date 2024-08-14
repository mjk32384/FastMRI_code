# SSIM mask 형태를 png파일로 저장

import cv2
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

image_directory_list = [Path('../../home/Data/train'),
                   Path('../../home/Data/val'),
                   Path('../../home/Data/leaderboard/acc5'),
                   Path('../../home/Data/leaderboard/acc9')]

for directory in image_directory_list:
    image_files = sorted(list(Path(directory / "image").iterdir()))
    for image_dir in tqdm(image_files, desc=f'{str(directory).split("/",4)[-1]}'):
        with h5py.File(image_dir, "r") as hf:
            for dataslice in range(hf['image_label'].shape[0]):
                target = hf['image_label'][dataslice]
                mask = np.zeros(target.shape)
                mask[target>5e-5] = 1
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=15)
                mask = cv2.erode(mask, kernel, iterations=14)

                plt.imshow(target, cmap='gray')
                plt.imshow(mask, cmap='Reds', alpha = mask/2)
                plt.axis('off')
                plt.savefig(f'../image/{str(directory).split("/",4)[-1]}/{image_dir.name.split(".")[0]}_{dataslice}.png', bbox_inches='tight', pad_inches=0)
                plt.clf()
plt.close('all')