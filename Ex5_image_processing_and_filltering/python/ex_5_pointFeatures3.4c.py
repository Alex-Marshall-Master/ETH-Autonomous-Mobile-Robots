import cv2 as cv                        # install by "pip install opencv-python"
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from ex_5_pointFeatures_utils import insertMarker, getPointLists, \
    stitchImages, detectHarrisFeatures

## Autonomous Mobile Robots - Exercise 5 (Image Saliency)

# Q3.4.c) Panorama creation with scaled images
# Now that you have seen, that the Harris corner detection is invariant to
# rotation (in-plane), could we also stitch together the images if the
# right image was scaled? Have a look at the detected corners for the scaled
# right image and compare them with the ones found in the left image.
# If they are not the same, what type of other feature could you use?
lena = cv.imread('Ex5_image_processing_and_filltering\python\lena.jpeg', cv.IMREAD_GRAYSCALE)
left_lena = cv.imread('Ex5_image_processing_and_filltering\python\left_lena.jpg', cv.IMREAD_GRAYSCALE)
right_lena = cv.imread('Ex5_image_processing_and_filltering\python\scaled_lena.jpg', cv.IMREAD_GRAYSCALE)

# TODO: Extract the corners in the left and right image using the Harris
# corner detector
left_corners = detectHarrisFeatures(left_lena, blockSize=5, threshold=0.08)
right_corners = detectHarrisFeatures(right_lena, blockSize=5, threshold=0.08)

# visualize our detected corners
left_J = cv.cvtColor(left_lena.copy(), cv.COLOR_GRAY2RGB)
left_J = insertMarker(left_J, left_corners)
right_J = cv.cvtColor(right_lena.copy(), cv.COLOR_GRAY2RGB)
right_J = insertMarker(right_J, right_corners, marker_size = 7)
fig11, ax11 = plt.subplots(num='Left corners marked for scaled right image')
plt.imshow(left_J, cmap='gray')
fig12, ax12 = plt.subplots(num='Right corners marked for scaled right image')
plt.imshow(right_J, cmap='gray')

# TODO: Could we also stitch together the images if the right image is scaled?


# TODO: If not, what type of other feature could you we use to solve this problem?

sift = cv.SIFT_create()
left_points, left_features = sift.detectAndCompute(left_lena, None)
right_points, right_features = sift.detectAndCompute(right_lena, None)

bf = cv.BFMatcher(crossCheck=True)
feature_pairs = bf.match(right_features, left_features)

feature_pairs = sorted(feature_pairs, key = lambda x:x.distance)
right_matched_points, left_matched_points = getPointLists(feature_pairs, right_points, left_points)

robust_right_matched = right_matched_points[:70]
robust_left_matched = left_matched_points[:70]
right_left_transform, inliers = cv.estimateAffinePartial2D(robust_right_matched, robust_left_matched)
left_left_transform = np.eye(3)
left_left_transform = left_left_transform[:2,:]

right_lena_color = right_lena.copy()
right_lena_color = cv.applyColorMap(right_lena_color, cv.COLORMAP_AUTUMN)
left_lena_color = left_lena.copy()
left_lena_color = cv.applyColorMap(left_lena_color, cv.COLORMAP_MAGMA)

img_size = [lena.shape[0], lena.shape[1], 3]         # we overlay color images, therefore our third component is 3 instead of 1
transforms = [left_left_transform, right_left_transform]
images = [left_lena_color, right_lena_color]
full_lena = stitchImages(transforms, images, img_size)

fig4c, ax4c = plt.subplots(num='Q3.4c Image alignment, scale - Stitched image')
ax4c.imshow(full_lena)

plt.show()
