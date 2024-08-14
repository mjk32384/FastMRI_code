import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, acc_weight = None, default_acc = False, validate=False, acc=None, mask_mode='equispaced'):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.acc_weight = acc_weight
        self.default_acc = default_acc
        self.validate = validate
        self.acc = acc
        # added here
        self.mask_mode = mask_mode
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            # validate인 경우에 모든 데이터를 다 쓰면 너무 오래 걸리니 일부만 사용
            # shuffle을 추가할 수도 있는데, seed 고정을 해도되는지 몰라서 일단 pass
            if self.validate:
                # sorted 추가 (validation을 동일한 dataset으로 하기 위해)
                image_files = sorted(image_files)[:len(image_files)//5]
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        # validate인 경우에 모든 데이터를 다 쓰면 너무 오래 걸리니 일부만 사용
        # shuffle을 추가할 수도 있는데, seed 고정을 해도되는지 몰라서 일단 pass
        if self.validate:
            # sorted 추가 (validation을 동일한 dataset으로 하기 위해)
            kspace_files = sorted(kspace_files)[:len(kspace_files)//5]
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

        # edited from here
        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            # reconstruct_json.py을 실행했을 때에는 default_acc = True
            if self.default_acc:
                mask = np.array(hf["mask"])
            # validate: validation하는 경우에 acc_weight과는 다른 acc를 주기 위해 추가함
            elif self.validate:
                mask = create_mask({self.acc: 1}, hf["kspace"].shape[-1], 'equispaced')
            else:
                mask = create_mask(self.acc_weight, hf["kspace"].shape[-1], self.mask_mode)
        # to here
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


# create_data_loader가 validation/test시에도 문제 없는지 한번 확인해야 할 듯
# isforward: reconstruction 할 때만 true
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
        acc = acc,
        # added here
        mask_mode = args.mask_mode
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader

# edited from here
def create_mask(acc_weight, width, mask_mode):
    acc = random.choices(list(acc_weight.keys()), weights=acc_weight.values(), k=1)[0]

    if mask_mode == 'equispaced':
        if width == 392:
            mask = np.array([int((i-196)%acc == 0) for i in range(392)], dtype=np.float32)
            mask[181:212] = 1
        elif width == 396:
            mask = np.array([int((i-198)%acc == 0) for i in range(396)], dtype=np.float32)
            mask[182:214] = 1
        else:
            raise Exception("Invalid Mask Width")
    elif mask_mode == 'random':
        if width == 392:
            mask = (np.random.rand(392)<1/acc).astype(np.float32)
            mask[181:212] = 1
        elif width == 396:
            mask = (np.random.rand(396)<1/acc).astype(np.float32)
            mask[182:214] = 1
    else:
        raise Exception("Invalid Mask Mode")
    
    return mask
# to here