import argparse, os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 120560400
import PIL.Image as pil_image
import rasterio as rio

from model import SRCNN, Subpixel
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, convert_TIFF_JPG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--model', type=str, default="SRCNN")
    parser.add_argument('--test-data', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model to device
    if (args.model == "SRCNN"):
        model = SRCNN().to(device)
    elif (args.model == "Subpixel"):
        model = Subpixel(upscale_factor=args.scale).to(device)

    # Load weights
    '''state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)'''

    # print(torch.load(args.weights_file))

    model.load_state_dict(torch.load(args.weights_file))

    model.eval()

    if not args.test_data:
        all_images = os.listdir('../{}'.format(args.image_dir))
        os.chdir('../{}'.format(args.image_dir))
    else:
        all_images = os.listdir(args.image_dir)
        os.chdir(args.image_dir)
        print(all_images)

    for image_file in all_images:
        if not ".DS_Store" in image_file:
            # Load and normalise image
            if "tiff" in image_file:
                image = rio.open(r"{}".format(image_file), count=3)
                band1 = image.read(1).astype(np.float32)
                band1 /= band1.max() / 255.0
                band2 = image.read(2).astype(np.float32)
                band2 /= band2.max() / 255.0
                band3 = image.read(3).astype(np.float32)
                band3 /= band3.max() / 255.0
                image = np.array((band1, band2, band3)).astype(np.float32)
                image = np.transpose(image, (1, 2, 0))  # else image.shape = (3, n, n)
            else:
                image = pil_image.open(image_file).convert('RGB')

                # Their downscaling
                image_width = (image.width // args.scale) * args.scale
                image_height = (image.height // args.scale) * args.scale
                image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
                image = image.resize((image.width // args.scale, image.height // args.scale),
                                     resample=pil_image.BICUBIC)
                image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
                image.save(image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

                image = np.array(image).astype(np.float32)

            ycbcr = convert_rgb_to_ycbcr(image)

            y = ycbcr[..., 0]
            y /= 255.
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                print("predicting...")
                preds = model(y).clamp(0.0, 1.0)

            print("predicted!")
            psnr = calc_psnr(y, preds)
            print('PSNR: {:.2f}'.format(psnr))

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            image_file = image_file.replace('.jpg', '_{}_x{}.jpg'.format(args.model, args.scale))
            output.save(image_file.replace('.tiff', '_{}_x{}.jpg'.format(args.model, args.scale)))
