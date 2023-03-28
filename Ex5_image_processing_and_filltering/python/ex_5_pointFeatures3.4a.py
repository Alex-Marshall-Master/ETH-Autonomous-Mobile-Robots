import cv2 as cv                        # install by "pip install opencv-python"
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from ex_5_pointFeatures_utils import insertMarker, convertToKeypoints, \
    getPointLists, stitchImages, detectHarrisFeatures

## Autonomous Mobile Robots - Exercise 5 (Image Saliency)

## Q3.4 Feature matching for panorama creation
# The aim of this exercise is to understand how a feature detector
# and feature descriptor work. We will investigate how we can use them to stitch together
# images together with and without image modifications such as rotation.

# In the following, there are several lines of code that need to be completed. You can look
# at the opencv documentation to find necessary functions

## Q3.4.a) Panorama creation with partial images
# read images
lena = cv.imread('Ex5_image_processing_and_filltering\python\lena.jpeg', cv.IMREAD_GRAYSCALE)
left_lena = cv.imread('Ex5_image_processing_and_filltering\python\left_lena.jpg', cv.IMREAD_GRAYSCALE)
right_lena = cv.imread('Ex5_image_processing_and_filltering\python\\right_lena.jpg', cv.IMREAD_GRAYSCALE)
# visualize the image
fig1a, ax1a = plt.subplots(1, 2, num='Q3.4a - Image alignment, translation (original images)')
fig1a.set_size_inches(8, 5)
ax1a[0].imshow(left_lena, cmap='gray')
ax1a[0].set_title('Left image part')
ax1a[1].imshow(right_lena, cmap='gray')
ax1a[1].set_title('Right image part')

# As a next step we need to find keypoints in both images. We will do this
# by identifying corners using the Harris corner detection that you have learned
# about in the lecture. The corner detector should recognize the same
# keypoints in both images where there is overlap.
# TODO: Extract the corners in the left and right image using the Harris
# corner detector (detectHarrisFeatures) given in the _extras file (this is
# essentially a wrapper around the cv.cornerHarris function)
# We found for this question that blocksize=5, ksize=3, k=0.04, and
# corner threshold=0.1 worked well. You can try modifying these values.
left_corners = detectHarrisFeatures(left_lena)
right_corners = detectHarrisFeatures(right_lena)

left_J = cv.cvtColor(left_lena.copy(), cv.COLOR_GRAY2RGB)
left_J = insertMarker(left_J, left_corners)
fig2a, ax2a = plt.subplots(1, 2, num='Q3.4a Image alignment, translation - Harris corners')
fig2a.set_size_inches(8, 5)
ax2a[0].imshow(left_J, cmap='gray')
ax2a[0].set_title('Left corners marked')

right_J = cv.cvtColor(right_lena.copy(), cv.COLOR_GRAY2RGB)
right_J = insertMarker(right_J, right_corners)
ax2a[1].imshow(right_J, cmap='gray')
ax2a[1].set_title('Right corners marked')

# After identifying keypoints, we will need a suitable feature descriptor
# for feature matching between the image pairs.

# convert the found corners to the cv.KeyPoint type expected by SIFT
left_corners = convertToKeypoints(left_corners.astype(float))
right_corners = convertToKeypoints(right_corners.astype(float))

# TODO: Use the obtained corners to extract SIFT feature descriptors
# NOTE - Once created, the sift object has a 'compute' method to generate
# descriptors (https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
sift = cv.SIFT_create()
left_points, left_features = sift.compute(left_lena, left_corners)
right_points, right_features = sift.compute(right_lena, right_corners)

# As a last step, we need to match the features found in the left and the
# right image. If this succeeds, we will be able to properly overlay the
# two images
# TODO: Find matching features between the left and the right image.

# create feature matcher - we will use the brute force matcher
bf = cv.BFMatcher(crossCheck=True)
# match features (matcher has a 'match method,
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
feature_pairs = bf.match(right_features, left_features)
right_matched_points, left_matched_points = getPointLists(feature_pairs, right_points, left_points)

# Let's visualize our matches.
matched_img = cv.drawMatches(right_lena, right_points, left_lena, left_points, feature_pairs, left_lena)
fig3a, ax3a = plt.subplots(num='Q3.4a Image alignment, translation - Feature matches')
ax3a.imshow(matched_img)

# These matched pairs look good, so let's align the images! For this to
# happen we need to compute the transformation between the left and the
# right image.
# TODO: Use the function "estimateAffinePartial2D" and the found feature_pairs
# to estimate the transformation between the left and the right image.
right_left_transform, inliers = cv.estimateAffinePartial2D(right_matched_points, left_matched_points)

# Convert images to color images for better visualization of the overlay
right_lena_color = right_lena.copy()
right_lena_color = cv.applyColorMap(right_lena_color, cv.COLORMAP_AUTUMN)
left_lena_color = left_lena.copy()
left_lena_color = cv.applyColorMap(left_lena_color, cv.COLORMAP_MAGMA)

# The function stichImages that will take care of overlaying the images for
# us needs as input the images, the transformations of these images with
# respect to the first image as well as the size of the final panorama.
# TODO: State the transformation for the first image, the left image? (State it in the form R | t)
left_left_transform = np.eye(3)[:2,:]

# create expected inputs for the function
img_size = [lena.shape[0], lena.shape[1], 3]         # we overlay color images, therefore our third component is 3 instead of 1
transforms = [left_left_transform, right_left_transform]
images = [left_lena_color, right_lena_color]
full_lena = stitchImages(transforms, images, img_size)

# Let's see our result
# figure('Name','Stitched images')
fig4a, ax4a = plt.subplots(num='Q3.4a Image alignment, translation - Stitched images')
ax4a.imshow(full_lena)

# Now,let's take a look at the estimated transform.
print('Rotation:', right_left_transform[:,:2])
print('Translation:', right_left_transform[:,2])

# TODO: Based on the knowledge that the two images were created by cutting out pieces from the original
# image without shifting it up or down, scale or rotation, did we achieve a
# perfect alignment using our keypoints? If not, what caused this?

plt.show()
