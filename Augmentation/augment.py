import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import h5py

from tqdm import tqdm

import PIL.Image as pil_image
import rasterio as rio

Image.MAX_IMAGE_PIXELS = 120560400

from model import SRCNN, Subpixel, FSRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, convert_TIFF_JPG


def prepare(args):
    h5_file = h5py.File(args.hdf5_file, 'w')  # args.hdf5_file

    label = h5_file.create_group('label')
    # TODO CREATE INFO

    all_images = os.listdir('../{}'.format(args.satellite_image_dir))
    os.chdir('../{}'.format(args.satellite_image_dir))

    for i, image_file in tqdm(enumerate(sorted(all_images))):
        if not ".DS_Store" in image_file:
            image = rio.open(r"{}".format(image_file), count=3)
            band1 = image.read(1).astype(np.float32)
            band1 /= band1.max() / 255.0
            band2 = image.read(2).astype(np.float32)
            band2 /= band2.max() / 255.0
            band3 = image.read(3).astype(np.float32)
            band3 /= band3.max() / 255.0
            image = np.array((band1, band2, band3)).astype(np.float32)
            image = np.transpose(image, (1, 2, 0))  # else image.shape = (3, n, n)
            ycbcr = convert_rgb_to_ycbcr(image)

            label.create_dataset(i, data=ycbcr)

    h5_file.close()


def augment(args, dataset):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model to device
    if args.model == "SRCNN":
        model = SRCNN().to(device)
    elif args.model == "Subpixel":
        model = Subpixel().to(device)
    elif args.model == "FSRCNN":
        model = FSRCNN().to(device)

    # Load weights
    state_dict = model.state_dict()
    for n, p in torch.load(args.augmentation_weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    i = 0
    for data in dataset :
        y = data[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, data[..., 1], data[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save('augmented_{}.jpg'.format(i)) #tiff or jpg
        i = i+1
