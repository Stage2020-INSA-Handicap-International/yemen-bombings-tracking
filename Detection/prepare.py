import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pl
from tqdm import tqdm


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    gt_group = h5_file.create_group('gt')
    labels_group = h5_file.create_group('label')

    for i, path in tqdm(enumerate(sorted(glob.glob('{}/*'.format(args.images_dir))))):
        img = pl.open(path).convert('RGB')
        # img = img.resize((4800, 4800), pl.ANTIALIAS)
        gt_group.create_dataset(str(i), data=np.array(img))

    for i, path in tqdm(enumerate(sorted(glob.glob('{}/*'.format(args.label_dir))))):
        labels_group.create_dataset(str(i), data=np.loadtxt(path).reshape(-1, 5))

    h5_file.create_dataset('info', data="detection")

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default="data/alpha/train")
    parser.add_argument('--label-dir', type=str, default="data/alpha/train_labels")
    parser.add_argument('--output-path', type=str, default="dataset.h5")
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if not args.train:
        eval(args)
    else:
        train(args)
