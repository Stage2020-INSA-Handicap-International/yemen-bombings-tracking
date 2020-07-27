function [Y] = untitled3(inputArg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Y = rgb2gray(inputArg);
%Y = histeq(I);
%Y=255-Y;
%Y = watershed(Y);
%Y = label2rgb(Y,'jet',[.5 .5 .5]);
Y = kuwahara(im2double(Y),21);
im1=Y;
%Y=imbinarize(Y,'adaptive','ForegroundPolarity','bright','Sensitivity',0.5);
%Y=entropyfilt(Y,true(3));
%Y = histeq(Y);
Y=imbinarize(Y,'adaptive','ForegroundPolarity','bright','Sensitivity',0.35);
im2=Y;
Y=medfilt2(Y,[25 25]);
im3=Y;
%Y=entropyfilt(Y,true(21));
im4=Y;
figure
montage(cat(3,im1,im2,im3,im4))
%Y=medfilt2(Y,[20 20]);
end


