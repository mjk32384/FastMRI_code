from graph import parse, SSIM_by_slice, flatten_list
import matplotlib.pyplot as plt
import numpy as np
import pickle, h5py
with open('test/ssim_log.pkl', 'rb') as f:
    ssim_log = pickle.load(f)[0]
data_path_val = '../../home/Data/val/'
ssim_log_396 = {}
ssim_log_392 = {}
for key, value in ssim_log.items():
    with h5py.File(data_path_val+'/kspace/'+key) as f:
        if f['kspace'].shape[-1] == 396:
            ssim_log_396[key] = value
        elif f['kspace'].shape[-1] == 392:
            ssim_log_392[key] = value
        else:
            print("Error!")
ssim_values_392 = np.array(flatten_list([list(dictionary.values()) for dictionary in ssim_log_392.values()]))
ssim_values_396 = np.array(flatten_list([list(dictionary.values()) for dictionary in ssim_log_396.values()]))
plt.hist(ssim_values_396, bins = 20, label = 'kspace width: 396')
plt.hist(ssim_values_392, bins = 20, label = 'kspace width: 392')
plt.xlabel('SSIM loss')
plt.ylabel('# of data')
plt.legend()
from graph import SSIM_by_pixel, parse, reconstruction_slice
import matplotlib.pyplot as plt
import random, torch

huge_loss_files = [[filename, slice] for filename, dictionary in ssim_log_396.items() for slice, ssim in dictionary.items() if ssim > 0.035]
random.shuffle(huge_loss_files)

sample = 4
fig, ax = plt.subplots(sample, 3, figsize = (3*5, sample*5))

for i, [filename, slice] in enumerate(huge_loss_files[:sample]):
    
    netname = 'test_6125_all_ssim_mask_40'
    image_fname = '../../home/Data/val/image/' + filename
    kspace_fname = '../../home/Data/val/kspace/' + filename

    mask_mode = 'equispaced'
    mask_acc = 9

    args = parse(['-n', netname, '--mask_mode', mask_mode, '--acc_weight', f'{{"{mask_acc}":"1"}}'])
    args.acc_weight = {int(k):int(v) for k, v in args.acc_weight.items()}
    args.image_fname = image_fname
    args.kspace_fname = kspace_fname
    args.dataslice = slice
    args.use_SSIM_mask = True

    ssim_graph = SSIM_by_pixel(args)
    

    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    recon, target, _ = reconstruction_slice(args, device)
    ax[i, 1].imshow(target.squeeze().cpu(), cmap = 'gray')
    ax[i, 1].set_xlabel('original')
    ax[i, 2].imshow(recon.squeeze().cpu(), cmap = 'gray')
    ax[i, 2].set_xlabel('reconstruction')

    ax[i, 0].imshow(ssim_graph, cmap='gray')
    ax[i, 0].set_xlabel('SSIM')
    ax[i, 1].set_title(f"mean SSIM: {ssim_graph.mean():.4f}")

for i in range(sample*3):
    ax[i//3, i%3].axis('off')