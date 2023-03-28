function color_img = getColorImage(img, hue_value)
% This function converts a grayscale image to a color image based on the
% example given at https://ch.mathworks.com/matlabcentral/answers/43977-colorizing-grayscale-images-with-tones-of-specified-colors

RGB = cat(3, img, img, img);
HSV = rgb2hsv(RGB);
HSV(:,:,1) = hue_value/360;  %Change the "hue" of the image, norm to 360 degrees
HSV(:,:,2) = 1 - HSV(:,:,2); %invert the saturation axis
color_img = hsv2rgb(HSV);
end