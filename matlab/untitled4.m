function [Y] = untitled4(inputArg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
YR = inputArg(:,:,1);
YB = inputArg(:,:,2);
im1=YR;
im2=YB;
YR=imbinarize(YR,0.95);
YB=imbinarize(YB,0.95);
im3=YR;
im4=YB;
SE=strel("disk",10);
%YR=imerode(YR,SE);
%YB=imerode(YB,SE);
im5=YR;
im6=YB;
figure
montage(cat(3,im1,im2,im3,im4,im5,im6))
figure
Y=cat(3,YR,YB,YB);
Y=double(Y);
end


