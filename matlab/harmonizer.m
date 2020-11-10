function [src] = harmonizer(src,target)
%HARMONIZER Summary of this function goes here
%   Detailed explanation goes here
src=imhistmatch(src,target);
end

