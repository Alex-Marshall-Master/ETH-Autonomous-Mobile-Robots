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

## Q3.4.b) Panorama creation with rotated images
# Now that you have seen the workflow for stitching the images together,
# let's try whether this also works in case we have a rotated second image.
lena = cv.imread('Ex5_image_processing_and_filltering\python\lena.jpeg', cv.IMREAD_GRAYSCALE)
left_lena = cv.imread('Ex5_image_processing_and_filltering\python\left_lena.jpg', cv.IMREAD_GRAYSCALE)
right_lena = cv.imread('Ex5_image_processing_and_filltering\python\\rotated_lena.jpg', cv.IMREAD_GRAYSCALE)
# visualize the image
# visualize the image
fig1b, ax1b = plt.subplots(1, 2, num='Q3.4b Image alignment, rotation - Original')
fig1b.set_size_inches(8, 5)
ax1b[0].imshow(left_lena, cmap='gray')
ax1b[0].set_title('Left image part')
ax1b[1].imshow(right_lena, cmap='gray')
ax1b[1].set_title('Right image part (rotated)')

# TODO: Implement the pipeline as introduced previously for the rotation case.
#  Use only the 70 best feature pairs after matching in this case to achieve a
#  more robust transformation estimate.

# 1. Corner detection using the Harris corner detector
# We found for this question that blocksize=4, ksize=3, k=0.04, and
# corner threshold=0.06 worked well. You can try modifying these values.
left_corners = detectHarrisFeatures(left_lena, blockSize=4, threshold=0.06)
right_corners = detectHarrisFeatures(right_lena, blockSize=4, threshold=0.06)

# 2. SIFT feature extraction
# convert the found corners to the Keypoint type expected by SIFT
left_corners = convertToKeypoints(left_corners.astype(float))
right_corners = convertToKeypoints(right_corners.astype(float))
sift = cv.SIFT_create()
left_points, left_features = sift.compute(left_lena, left_corners)
right_points, right_features = sift.compute(right_lena, right_corners)

# 3. Feature matching
# create feature matcher
bf = cv.BFMatcher(crossCheck=True)
# match features
feature_pairs = bf.match(right_features, left_features)
# Sort them in the order of their distance to extract best matches
feature_pairs = sorted(feature_pairs, key = lambda x:x.distance)
right_matched_points, left_matched_points = getPointLists(feature_pairs, right_points, left_points)

# Let's visualize our matches.
matched_img = cv.drawMatches(right_lena, right_points, left_lena, left_points, feature_pairs[:70], left_lena)
fig3b, ax3b = plt.subplots(num='Q3.4b Image alignment, rotation - Feature matches')
ax3b.imshow(matched_img)

# 4. Relative transformation based on feature pairs
# We use only the first best 70 pairs in this case to get a more robust estimate
robust_right_matched = right_matched_points[:70]
robust_left_matched = left_matched_points[:70]
right_left_transform, inliers = cv.estimateAffinePartial2D(robust_right_matched, robust_left_matched)

# identity matrix for left to left image transform
left_left_transform = np.eye(3)[0:2,:]

# Convert images to color images for better visualization of the overlay.
right_lena_color = right_lena.copy()
right_lena_color = cv.applyColorMap(right_lena_color, cv.COLORMAP_AUTUMN)
left_lena_color = left_lena.copy()
left_lena_color = cv.applyColorMap(left_lena_color, cv.COLORMAP_MAGMA)

# 5. Stitch together images and save them in result image called full_lena
img_size = [lena.shape[0], lena.shape[1], 3]
transforms = [left_left_transform, right_left_transform]
images = [left_lena_color, right_lena_color]
full_lena = stitchImages(transforms, images, img_size)

# Let's see our result
fig4b, ax4b = plt.subplots(num='Q3.4b Image alignment, rotation - Stitched image')
ax4b.imshow(full_lena)

plt.show()
