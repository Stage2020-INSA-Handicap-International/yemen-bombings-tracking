import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
import cv2
from tqdm import tqdm


def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    images = []
    labels = []
    for path in tqdm(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        if "jpg" in path :
            img = cv2.imread(path)
            images.append(img)
        elif "txt" in path :

    h5_file.create_dataset('img', data=images)
    h5_file.create_dataset('label', data=)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=10)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)