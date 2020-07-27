function [I] = untitled5(inputArg,template,angle,step,cutoff)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
I = double(inputArg);
% Show image
imshow(I)
% You have to select a part of single occurrence of the pattern (a template) on the image! See below image.
%rect = round(getrect);
% In case it is a multiband image make grayscale image
if size(I,3)>1
    BW = rgb2gray(I);
else
    BW = I;
end
% Extract template from BW
%template = BW(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:);
% Show template - this is the extent you selected during "getrect"
template=rgb2gray(imread(template));
imshow(template)
for Ang=0:step:angle % rotate the template Itm 1 degree at the time and look for it in the image Is
    Itr=Rotate_binary_edge_image(template,Ang);
    disp([num2str((Ang)/3.6) '% Scanned' ' Ang: ' num2str(Ang) '°']);
    % Calculate how much said template correlates on each pixel in the image
    C = normxcorr2(Itr,BW);
    % Remove padded borders from correlation
    pad = floor(size(Itr)./2);
    center = size(I);
    C = C([false(1,pad(1)) true(1,center(1))], ...
            [false(1,pad(2)) true(1,center(2))]);
    % Plot the correlation
    %figure, surf(C), shading flat

    % Get all indexes where the correlation is high. Value read from previous figure.
    % The lower the cut-off value, the more pixels will be altered
    idx = C>cutoff;
    % Dilate the idx because else masked area is too small
    idx = imdilate(idx,strel('disk',1));
    % Replicate them if multiband image. Does nothing if only grayscale image
    idx = repmat(idx,1,1,size(I,3));
    % Replace pattern pixels with NaN
    I(idx) = NaN;
    % Fill Nan values with 4x4 median filter
    I = fillmissing(I,'constant',0);
    % Display new image
end
figure; imshow(I)
end

