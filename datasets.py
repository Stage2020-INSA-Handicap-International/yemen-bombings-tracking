import numpy as np
from torch.utils.data import Dataset
import h5py
import glob
import cv2
from PIL import Image

class HDF5torch(Dataset):
    def __init__(self, h5_file):
        super(HDF5torch, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            #if f['info'][str(1)] == 'unprocessed':
            return f['label'][str(idx)][:, :], f['target'][str(idx)][:, :]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['label'])

# Could create a tensorflow dataset aswell