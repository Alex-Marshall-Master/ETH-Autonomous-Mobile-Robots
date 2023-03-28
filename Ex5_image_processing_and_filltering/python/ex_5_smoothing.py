## Autonomous Mobile Robots - Exercise 5 (Image Saliency)
import numpy as np
import cv2

## Q1.2.c Smoothing in 2D
# The aim of this exercise is understanding how filtering works in 2
# dimensions. To this end, your task is to read and understand the MATLAB
# code provided for smoothing an image and then fill in the missing
# expressions, which will permit the application of the filter to the
# correct portion of the original image.

# We will be working with the image of Lena that you know from the lecture.
# Note - to close the windows click any key.

# read image
lena = cv2.imread('Ex5_image_processing_and_filltering\python\lena.jpeg', cv2.IMREAD_GRAYSCALE)
# visualize the image
hFig1 = cv2.imshow("Original Lena", lena)

# get the dimensions
height = lena.shape[0]
width = lena.shape[1]

# output image
filtered_lena = np.zeros_like(lena)

# The filter used for smoothing is the following kernel.
F = np.array([[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]], dtype=float) / 256.0

# In the lecture, different ways of handling the boundaries of an image
# when applying the filter were discussed. In this exercise we are ignoring
# the boundaries of the original image, by setting the corresponding
# filtered image values to zero.

# compute the offset for the rows and the columns, where the original image
# boundaries are to be ignored
row_o = F.shape[0] // 2
col_o = F.shape[1] // 2

# loop through all the pixels of the filtered image that we need to set their value
# (i.e. ignoring the pixels at the image boundaries)
#
#   TODO:  DEFINE THE LIMITS FOR THE ROW (i) AND COLUMN (j) INDICES
# these are the maximum indices for row and column where the filter will be
# placed
row_limit = height - row_o
col_limit = width - col_o

for i in range(row_o, row_limit):
    for j in range(col_o, col_limit):
        ##################################################################
        ##     TODO: FILL IN THE EXPRESSIONS                            ##
        # Extract the relevant original-image values (to be stored in    #
        # tempLena) which will be multiplied by the filter values. In    #
        # order to do this, you need to fill in the expressions for the  #
        # row and column limits of the values to be extracted from lena  #
        # Given your current filter center (i,j) what are the indices of #
        # the left/right/top/bottom filter edges.
        row_min = i - row_o
        row_max = i + row_o
        col_min = j - col_o
        col_max = j + col_o
        ##     END OF YOUR INPUT                                        ##
        ##################################################################
        # Note that we add the +1, so if row_min = 0, row_max = 4, we will
        # return elements 0, 1, 2, 3, 4
        tempLena = lena[row_min:(row_max+1), col_min:(col_max+1)]
        filtered_lena[i, j] = (F * tempLena).sum()

diff_image = abs(lena.astype(int)-filtered_lena.astype(int)).astype('uint8')
cv2.imshow('Difference (original - filtered)', diff_image)

# Note that the differences are quite small magnitude, so we can also multiply
# then threshold to see the difference a little more clearly
cv2.imshow('Difference (original - filtered)*3, threshold',
           np.minimum(255, diff_image*3))

# display the filtered image
cv2.imshow('Filtered (smoothed) image', filtered_lena)

cv2.waitKey(0)
# Click any key to close the windows
cv2.destroyAllWindows()
