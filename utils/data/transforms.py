import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mraugment.data_augment import DataAugmentor
import argparse

# 여기는 그냥 거의 다 새로 짬
# to_tensor()함수는 그냥 지움

class DataTransform:
    def __init__(self, isforward, max_key, epoch_fn, augment, add_gaussian_noise):
        self.isforward = isforward
        self.max_key = max_key
        self.augment = augment
        self.add_gaussian_noise = add_gaussian_noise
        if self.augment:
            self.parser = argparse.ArgumentParser()
            # 여기서 Data Augmentation parameter 변경 가능
            self.parser = DataAugmentor.add_augmentation_specific_args(self.parser)
            self.parser.set_defaults(aug_on=True)
            self.args = self.parser.parse_args([])
            self.augmentor = DataAugmentor(self.args, epoch_fn)
        
    def __call__(self, mask, input, target, attrs, fname, slice):

        if not self.isforward:
            target = torch.from_numpy(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = torch.from_numpy(input)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        if self.augment:
            kspace, target = self.augmentor(kspace, kspace.shape[-3:-1], target)
            target = TF.center_crop(target, (384, 384))

        if self.add_gaussian_noise:
            target = self.image_add_gaussian_noise(target)

        kspace = kspace * mask
        

        return mask, kspace, target, maximum, fname, slice

    def image_add_gaussian_noise(self, image):
        """
        mean: 0, std: sqrt(median of window size = 11x11) * sigma
        """
        kernel_size = 11
        sigma = 0.03
        maximum = image.max()
        image = image/image.max()
        padding = kernel_size // 2
        unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), padding = padding, kernel_size=(kernel_size, kernel_size))

        unfolded = unfolded.reshape(kernel_size * kernel_size, -1)
        median_values = unfolded.median(dim=0).values

        sqrt_median_values = torch.sqrt(median_values).reshape(image.shape[0], image.shape[1])
        image = image + torch.normal(0, std=sigma*sqrt_median_values).numpy()
        
        image = image * maximum
        return image