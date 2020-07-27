function [out] = untitled2(src,target)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
out=cat(3,target-src,src-target,src-target);
out=imgaussfilt(out,5);
figure
%out=imfilter(out,ones([100 100])/100.^2);
end

