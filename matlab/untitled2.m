function [out] = untitled2(src,target)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
out=cat(3,target-src,src-target,src-target);
%out=double(imbinarize(out,0.3));
outR=out(:,:,1);
outB=out(:,:,2);

%R0.25 B0.5 ou R&B0.2
outR=imbinarize(outR,0.20);
outB=imbinarize(outB,0.3);

outR=medfilt2(outR,[25 25]);
outB=medfilt2(outB,[25 25]);
out=double(cat(3,outR,outB,outB));
%out=imgaussfilt(out,5);
figure
imshow(out)
%out=imfilter(out,ones([100 100])/100.^2);
end

