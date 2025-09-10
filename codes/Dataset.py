from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
OUT_DIR = Path('data')

class CrackSegDataset(Dataset):
    def __init__(self, split: str, mean: Tuple[float,float,float]=None, std: Tuple[float,float,float]=None):
        self.split = split
        self.im_dir = OUT_DIR / split / 'images_np'
        self.mask_dir = OUT_DIR / split / 'masks_np'
        self.paths = sorted([p for p in self.im_dir.glob('*.npy')])
        self.mean = mean
        self.std = std
        if len(self.paths) == 0:
            raise RuntimeError(f"No preprocessed npy files found in {self.im_dir}. Run preprocess.py first.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        x = np.load(p)            # HxWx3 float32 in 0..1
        y = np.load(self.mask_dir / p.name)  # HxW int64 in {0,1}

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        x = torch.from_numpy(x.transpose(2,0,1)).float()
        y = torch.from_numpy(y).long()
        return x, y

def make_dataloader(split: str, batch_size: int=4, shuffle: bool=True, mean=None, std=None, num_workers: int=0):
    ds = CrackSegDataset(split, mean=mean, std=std)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# make_dataloader('train', batch_size=4, shuffle=True, num_workers=2)