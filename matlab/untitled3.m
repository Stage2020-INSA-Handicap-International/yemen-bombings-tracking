function [Y] = untitled3(inputArg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Y = inputArg;
Y=rgb2hsv(Y);
%Y = rgb2gray(inputArg);
%Y = histeq(I);
%Y=255-Y;
%Y = watershed(Y);
%Y = label2rgb(Y,'jet',[.5 .5 .5]);
Yr=Y(:,:,1);
Yg=Y(:,:,2);
Yb=Y(:,:,3);
Yr = kuwahara(im2double(Yr),21);
Yg = kuwahara(im2double(Yg),21);
Yb = kuwahara(im2double(Yb),21);
Y=cat(3,Yr,Yg,Yb);
im1=Y;
%Y=imbinarize(Y,'adaptive','ForegroundPolarity','bright','Sensitivity',0.5);
%Y=entropyfilt(Y,true(3));
%Y = histeq(Y);
%Y=imbinarize(Y,'adaptive','ForegroundPolarity','bright','Sensitivity',0.35);
im2=Y;
%Y=medfilt2(Y,[25 25]);
im3=Y;
%Y=entropyfilt(Y,true(21));
im4=Y;
figure
montage(cat(3,im1,im2,im3,im4))
%Y=medfilt2(Y,[20 20]);
Y=Yb;
end


