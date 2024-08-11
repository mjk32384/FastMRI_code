import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

# class SliceData(Dataset):
#     def __init__(self, root, transform, input_key, target_key, forward=False):
#         self.transform = transform
#         self.input_key = input_key
#         self.target_key = target_key
#         self.forward = forward
#         self.image_examples = []
#         self.kspace_examples = []

#         if not forward:
#             image_files = list(Path(root / "image").iterdir())
#             for fname in sorted(image_files):
#                 num_slices = self._get_metadata(fname)

#                 self.image_examples += [
#                     (fname, slice_ind) for slice_ind in range(num_slices)
#                 ]

#         kspace_files = list(Path(root / "kspace").iterdir())
#         for fname in sorted(kspace_files):
#             num_slices = self._get_metadata(fname)

#             self.kspace_examples += [
#                 (fname, slice_ind) for slice_ind in range(num_slices)
#             ]


#     def _get_metadata(self, fname):
#         with h5py.File(fname, "r") as hf:
#             if self.input_key in hf.keys():
#                 num_slices = hf[self.input_key].shape[0]
#             elif self.target_key in hf.keys():
#                 num_slices = hf[self.target_key].shape[0]
#         return num_slices

#     def __len__(self):
#         return len(self.kspace_examples)

#     def __getitem__(self, i):
#         if not self.forward:
#             image_fname, _ = self.image_examples[i]
#         kspace_fname, dataslice = self.kspace_examples[i]

#         with h5py.File(kspace_fname, "r") as hf:
#             input = hf[self.input_key][dataslice]
#             mask =  np.array(hf["mask"])
#         if self.forward:
#             target = -1
#             attrs = -1
#         else:
#             with h5py.File(image_fname, "r") as hf:
#                 target = hf[self.target_key][dataslice]
#                 attrs = dict(hf.attrs)
            
#         return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
    
# def create_data_loaders(data_path, args, shuffle=False, isforward=False):
#     if isforward == False:
#         max_key_ = args.max_key
#         target_key_ = args.target_key
#     else:
#         max_key_ = -1
#         target_key_ = -1
#     data_storage = SliceData(
#         root=data_path,
#         transform=DataTransform(isforward, max_key_),
#         input_key=args.input_key,
#         target_key=target_key_,
#         forward = isforward
#     )

#     data_loader = DataLoader(
#         dataset=data_storage,
#         batch_size=args.batch_size,
#         shuffle=shuffle,
#     )
#     return data_loader


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, acc_weight = None, default_acc = False, validate=False, acc=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.acc_weight = acc_weight
        self.default_acc = default_acc
        self.validate = validate
        self.acc = acc
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            # validate인 경우에 모든 데이터를 다 쓰면 너무 오래 걸리니 일부만 사용
            # shuffle을 추가할 수도 있는데, seed 고정을 해도되는지 몰라서 일단 pass
            if self.validate:
                image_files = image_files[:len(image_files)//10]
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        # validate인 경우에 모든 데이터를 다 쓰면 너무 오래 걸리니 일부만 사용
        # shuffle을 추가할 수도 있는데, seed 고정을 해도되는지 몰라서 일단 pass
        if self.validate:
            kspace_files = kspace_files[:len(kspace_files)//10]
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.default_acc:
                mask = np.array(hf["mask"])
            elif self.validate:
                mask = create_mask({self.acc: 1}, hf["kspace"].shape[-1])
            else:
                mask = create_mask(self.acc_weight, hf["kspace"].shape[-1])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, default_acc=False, validate=False, acc=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        acc_weight = args.acc_weight,
        default_acc = default_acc,
        validate = validate,
        acc = acc
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader

def create_mask(acc_weight, width):
    acc = random.choices(list(acc_weight.keys()), weights=acc_weight.values())[0]
    if width == 392:
        mask = np.array([int((i-196)%acc == 0) for i in range(392)], dtype=np.float32)
        mask[181:212] = 1
    elif width == 396:
        mask = np.array([int((i-198)%acc == 0) for i in range(396)], dtype=np.float32)
        mask[182:214] = 1
    else:
        raise Exception("Invalid Mask Width")
    return mask