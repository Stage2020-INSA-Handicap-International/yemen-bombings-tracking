import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
from skimage import exposure
from skimage.transform import match_histograms
from skimage.filters import rank
from skimage.morphology import disk
from datasets import HDF5torch
from utils import histogram_equalization, image_difference
import tqdm

def preprocess(dataset) :
    i = 0
    for data in dataset:
        src, target = data
        if len(src.shape) > 2:
            src = histogram_equalization(src)
            target = histogram_equalization(target)
        else:
            src = cv2.equalizeHist(src)
            target = cv2.equalizeHist(target)

        matched = match_histograms(src, target, multichannel=False)
        # threshold up
        t_up = 255
        # threshold down
        t_down = 205
        dif, dif_placement = image_difference(matched, target, t_up, t_down)

        cv2.imwrite("{}/{}_dif_{}_{}.jpg".format(args.preprocessed_path, i, t_up,t_down), target * dif_placement)
        i = i+1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unprocessed-hdf5', type=str, default="unprocessed.h5")
    parser.add_argument('--preprocessed-path', type=str, default="data/preprocessed")
    args = parser.parse_args()

    dataset = HDF5torch(args.unprocessed_hdf5)
    preprocess(dataset)