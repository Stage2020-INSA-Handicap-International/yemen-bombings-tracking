function [Y] = untitled6(inputArg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
YR = inputArg(:,:,1);
YB = inputArg(:,:,2);
im1=YR;
im2=YB;
YR=imbinarize(YR,0.95);
YB=imbinarize(YB,0.95);
%YR=untitled5(YR,"untitled.png", 180,3,0.4);
%YB=untitled5(YB,"untitled.png", 180,3,0.4);
%YR=untitled5(YR,"dot.png", 1,3,0.5);
%YB=untitled5(YB,"dot.png", 1,3,0.5);
im3=YR;
im4=YB;
YR=medfilt2(YR,[25,25]);
YB=medfilt2(YB,[25,25]);
im5=YR;
im6=YB;
figure
montage(cat(3,im1,im2,im3,im4,im5,im6))
figure
Y=cat(3,YR,YB,YB);
Y=double(Y);
end
