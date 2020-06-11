import glob
import random
import os
import numpy as np
import h5py
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

##import matplotlib.pyplot as plt
##import matplotlib.patches as patches

from skimage.transform import resize

import sys
# TODO Detection Dataset
class DetectDataset(Dataset):
    def __init__(self, h5_file, img_size=4800, augment=True, multiscale=True, normalized_labels=True):
        super(DetectDataset, self).__init__()
        self.h5_file = h5_file
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, idx):
        #---------
        #  Image
        #---------
        with h5py.File(self.h5_file, 'r') as f:
            gt = np.expand_dims(f['gt'][str(idx)][:, :] / 255., 0)
            labels = np.expand_dims(f['label'][str(idx)][:, :] / 255., 0)

            img = transforms.ToTensor()(gt[0])

            # print(idx)
            # print(img.dim())
            # print("---")

            # Handle images with less than three channels
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))

            _, h, w = img.shape
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            # Pad to square resolution
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape

            # ---------
            #  Label
            # ---------

            boxes = torch.from_numpy(labels[0]).double()
            # print(boxes.dim())
            # print("---")

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            return img, targets

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['label'])

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets