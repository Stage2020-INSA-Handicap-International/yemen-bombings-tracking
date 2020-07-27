function [Y] = untitled(inputArg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
I = rgb2gray(inputArg);
Y = histeq(I);
%J = imnoise(I,'gaussian',0,0.005);
%Y = kuwahara(im2double(J),13);
%Y=imbinarize(Y,'adaptive','ForegroundPolarity','bright','Sensitivity',0.5);
figure
imshow(Y)
end


