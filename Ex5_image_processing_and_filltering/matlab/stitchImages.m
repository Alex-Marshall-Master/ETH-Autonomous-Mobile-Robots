function full_image = stitchImages(images, transforms, img_size)
% This function stitches together the input images based on the given
% transforms. It is based on this example: https://ch.mathworks.com/help/vision/ug/feature-based-panoramic-image-stitching.html

% Create a 2-D spatial reference object defining the size of the original
% lena image.
xLimits = [0 img_size(1)];
yLimits = [0 img_size(2)];
full_image = zeros(img_size);
panorama_view = imref2d([img_size(1) img_size(2)], xLimits, yLimits);

% Stich the two images together by transforming later images to the first image 
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Get the number of images 
num_imgs = size(images);

% Create the panorama.
for i = 1:num_imgs(2)
    
    I = images{i};   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, transforms{i}, 'OutputView', panorama_view);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), transforms{i}, 'OutputView', panorama_view);
    
    % Overlay the warpedImage onto the panorama.
    full_image = step(blender, full_image, warpedImage, mask);
end

end