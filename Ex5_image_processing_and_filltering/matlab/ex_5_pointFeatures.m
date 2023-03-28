%% Autonomous Mobile Robots - Exercise 5 (Image Saliency)
close all;
clear all;
clc;

%% Q3.4 Feature matching for panorama creation
% The aim of this exercise is to understand how a feature detector 
% and feature descriptor work. We will investigate how we can use them to stitch together 
% images together with and without image modifications such as rotation.

% In the following, there are several lines of code that need to be completed. You can look
% at the following Matlab examples to find the functions necessary: 
% https://ch.mathworks.com/help/vision/ug/local-feature-detection-and-extraction.html

%% Q3.4.a) Panorama creation with partial images
% read images
lena = imread('lena.jpeg');
left_lena = imread('left_lena.jpg');
right_lena = imread('right_lena.jpg');
% visualize the image 
figure('Name','Partial images')
subplot(1,2,1), imshow('left_lena.jpg')
subplot(1,2,2), imshow('right_lena.jpg')

% As a next step we need to find keypoints in both images. We will do this 
% by identifying corners using the Harris corner detection that you have learned 
% about in the lecture. The corner detector should recognize the same
% keypoints in both images where there is overlap.
% TODO: Extract the corners in the left and right image using the Harris
% corner detector.
left_corners = detectHarrisFeatures(left_lena);
right_corners = detectHarrisFeatures(right_lena);

left_J = insertMarker(left_lena, left_corners,'circle');
right_J = insertMarker(right_lena, right_corners,'circle');
figure('Name','Detected Harris corners')
subplot(1,2,1), imshow(left_J)
subplot(1,2,2), imshow(right_J)

% After identifying keypoints, we will need a suitable feature descriptor
% for feature matching between the image pairs. 
% TODO: Use your previously obtained corners to extract SIFT features. 
% (Hint: Use left_corners.Location to access the locations and avoid data type errors.)
[left_features, left_points] = extractFeatures(left_lena, left_corners.Location, 'Method', 'SIFT');
[right_features, right_points] = extractFeatures(right_lena, right_corners.Location, 'Method', 'SIFT');

% As a last step, we need to match the features found in the left and the
% right image. If this succeeds, we will be able to properly overlay the
% two images 
% TODO: Find matching features between the left and the right image.
feature_pairs = matchFeatures(left_features, right_features);

% Let's visualize our matches.
figure; ax = axes;
showMatchedFeatures(left_lena, right_lena, left_points(feature_pairs(1:70,1)), right_points(feature_pairs(1:70,2)), 'montage', 'Parent', ax);
title(ax,'Candidate point matches');
legend(ax,'Matched points 1','Matched points 2');

% These matched pairs look good, so let's align the images! For this to
% happen we need to compute the transformation between the left and the
% right image.
left_matched_points = left_points(feature_pairs(:,1));
right_matched_points = right_points(feature_pairs(:,2));

% TODO: Use the function "estimateGeometricTransform2D" and the found feature_pairs 
% to estimate the transformation between the left and the right image.
right_left_transform = estimateGeometricTransform2D(right_matched_points, left_matched_points, 'rigid');

% Convert images to color images for better visualization of the overlay
right_lena_color = getColorImage(right_lena, 163);
left_lena_color = getColorImage(left_lena, 50);

% The function stichImages that will take care of overlaying the images for
% us needs as input the images, the transformations of these images with 
% respect to the first image as well as the size of the final panorama. 
img_size = [size(lena,1) size(lena,2) 3];                % we overlay color images, therefore our third component is 3 instead of 1

% TODO: State the transformation for the first image, the left image?
left_left_transform = rigid2d;

images = {left_lena_color,right_lena_color};
transforms = {left_left_transform, right_left_transform};
full_lena = stitchImages(images, transforms, img_size);

% Let's see our result
figure('Name','Stitched images')
imshow(full_lena)

% Now,let's take a look at the estimated transform. 
disp('Rotation:');
disp(right_left_transform.Rotation);
disp('Translation:');
disp(right_left_transform.Translation);

% TODO: Based on the knowledge that the two images were created by cutting out piecies from the original
% image without shifting it up or down, scale or rotation, did we achieve a
% perfect alignment using our keypoints? If not, what caused this? 

%% Q3.4.b) Panorama creation with rotated images
% Now that you have seen the workflow for stitching the images together,
% let's try whether this also works in case we have a rotated second image.
right_lena = imread('rotated_lena.jpg');
% visualize the image 
figure('Name','Partial images for rotated right image')
subplot(1,2,1), imshow('left_lena.jpg')
subplot(1,2,2), imshow('rotated_lena.jpg')

% TODO: Implement the pipeline as introduced above for the rotation case. You can reuse
% your results from the previous steps for the left image.

% 1. Corner detection using the Harris corner detector 
right_corners = detectHarrisFeatures(right_lena);
right_J = insertMarker(right_lena, right_corners,'circle');
figure('Name','Detected Harris corners for rotated right image')
subplot(1,2,1), imshow(left_J)
subplot(1,2,2), imshow(right_J)

% 2. SIFT feature extraction
[right_features, right_points] = extractFeatures(right_lena, right_corners.Location, 'Method', 'SIFT');

% 3. Feature matching 
feature_pairs = matchFeatures(left_features, right_features);
figure; ax = axes;
showMatchedFeatures(left_lena, right_lena, left_points(feature_pairs(1:70,1)), right_points(feature_pairs(1:70,2)), 'montage', 'Parent', ax);
title(ax,'Candidate point matches for rotated right image');
legend(ax,'Matched points 1','Matched points 2');

% 4. Relative transformation based on feature pairs 
left_matched_points = left_points(feature_pairs(:,1));
right_matched_points = right_points(feature_pairs(:,2));
right_left_transform = estimateGeometricTransform2D(right_matched_points, left_matched_points, 'rigid');

% Color right image for better visualization
right_lena_color = getColorImage(right_lena, 163);

% 5. Stitch together images and save them in result image called full_lena
images = {left_lena_color,right_lena_color};
transforms = {left_left_transform, right_left_transform};
full_lena = stitchImages(images, transforms, img_size);

% Let's see our result
figure('Name','Stitched images for rotated right image')
imshow(full_lena)

%% Q3.4.c) Panorama creation with scaled images
% Now that you have seen, that the Harris corner detection is invariant to
% rotation (in-plane), could we also stitch together the images if the
% right image was scaled? Have a look at the detected corners for the scaled 
% right image and compare them with the ones found in the left image. 
% If they are not the same, what type of other feature could you use?

% TODO: Extract the corners in the left and right image using the Harris
% corner detector.
right_lena = imread('scaled_lena.jpg');
left_corners = detectHarrisFeatures(left_lena);
right_corners = detectHarrisFeatures(right_lena);

% visualize our detected corners
left_J = insertMarker(left_lena, left_corners,'circle');
right_J = insertMarker(right_lena, right_corners,'circle');
figure('Name','Detected Harris corners for scaled right image')
subplot(1,2,1), imshow(left_J)
subplot(1,2,2), imshow(right_J)

% TODO: Could we also stitch together the images if the right image is scaled?

% TODO: If not, what type of other feature could you we use to solve this problem?

