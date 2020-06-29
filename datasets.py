import numpy as np
from torch.utils.data import Dataset
import h5py
import glob
import cv2
from PIL import Image

class HDF5torch(Dataset):
    def __init__(self, h5_file):
        super(HDF5Dataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            if f['info'] == 'preprocess':
                return np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['label'])

# Could create a tensorflow dataset aswell