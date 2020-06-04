import numpy as np
from torch.utils.data import Dataset
import h5py
import glob
import cv2
from PIL import Image

''' Use prepare.py instead
def create_h5(created_file, my_directory="/data", name="dataset") :
    hf = h5py.File("{}/{}".format(my_directory,created_file), 'w')
    images = []
    scan_files=glob.glob("{}/*.jpg".format(my_directory))
    for imgfile in scan_files:
        img = cv2.imread(imgfile)
        images.append(img)
    hf.create_dataset(name,data=images)
    hf.close()
    '''
def data_print(h5file) :
    with h5py.File(h5file, 'r') as f :
        data = f.get('hr')
        print(data)

#Squelette de base a changer
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class ValDataset(Dataset):
    def __init__(self, h5_file):
        super(ValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class HDF5Dataset(Dataset):
    def __init__(self, h5_file):
        super(HDF5Dataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            #TODO CASE WITH INFO
            return f['label'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['label'])