import argparse
import h5py
from tqdm import tqdm
import os
import cv2


def prepare_preprocessing(args):
    h5_file = h5py.File(args.unprocessed_hdf5, 'w')  # args.hdf5_file

    label = h5_file.create_group('label')
    target = h5_file.create_group('target')
    info = h5_file.create_group('info')
    info.create_dataset(str(1), data="preprocess")
    # TODO CREATE INFO

    all_images = os.listdir('{}/'.format(args.unprocessed_path))
    os.chdir('{}/'.format(args.unprocessed_path))

    for image_file in tqdm(sorted(all_images)):
        if not ".DS_Store" in image_file:
            img = cv2.imread(image_file)
            i = image_file.split('_')[1].split('.')[0]
            if "src" in image_file:
                label.create_dataset(str(i), data=img)
            if "target" in image_file:
                target.create_dataset(str(i), data=img)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unprocessed-path', type=str, default="data/unprocessed")
    parser.add_argument('--unprocessed-hdf5', type=str, default="unprocessed.h5")
    parser.add_argument('--preprocess', action='store_true')
    args = parser.parse_args()

    if args.preprocess :
        prepare_preprocessing(args)
