import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
from skimage import exposure
from skimage.transform import match_histograms
from skimage.filters import rank
from skimage.morphology import disk

# replace with hdf5 dataset
file_name = 'sanaa_2'
src = cv2.imread('{}_052015.jpg'.format(file_name))
target = cv2.imread('{}_102015.jpg'.format(file_name))

if len(src.shape) > 2:
    src = histogram_equalization(src)
    target = histogram_equalization(target)
else:
    src = cv2.equalizeHist(src)
    target = cv2.equalizeHist(target)

matched = match_histograms(src, target, multichannel=True)

cv2.imwrite("{}_matched.jpg".format(file_name), matched)

# threshold up
t_up = 255
# threshold down
t_down = 205
dif, dif_placement = image_difference(matched, target, t_up, t_down)

cv2.imwrite("{}_dif_{}_{}.jpg".format(file_name, t_up,t_down), target * dif_placement)