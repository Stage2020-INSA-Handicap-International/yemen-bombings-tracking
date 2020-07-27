% clear all;close all;clc
im=imread('data/unprocessed/4_target.jpg');
% gray=rgb2gray(im);
imagBW = kapur(im);
imshow(imagBW)
figure;imshow(im)
[K] = otsu(gray); 
BW=im2bw(im,K/255);
imagBW = kapur(im);

subplot(121)
imshow(BW)
title('Otsu Threshold')
subplot(122)
imshow(imagBW)
title('Kapur Threshold')
