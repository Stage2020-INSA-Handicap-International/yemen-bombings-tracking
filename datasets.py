import numpy as np
import random
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, img_size=4800, augment=True, multiscale=True, normalized_labels=True):
        super(HDF5Dataset, self).__init__()
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            if f['info'] == "Detection" :
                self.img_size = img_size
                self.max_objects = 100
                self.augment = augment
                self.multiscale = multiscale
                self.normalized_labels = normalized_labels
                self.min_size = self.img_size - 3 * 32
                self.max_size = self.img_size + 3 * 32
                self.batch_count = 0

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            if f['info'] == 'Augmentation' :
                return np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)
            elif f['info'] == 'Detection' :
                return np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['label'])