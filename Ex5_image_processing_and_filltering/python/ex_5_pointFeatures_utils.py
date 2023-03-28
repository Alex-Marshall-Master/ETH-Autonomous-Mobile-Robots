import cv2 as cv                        # install by "pip install opencv-python"
import numpy as np

def detectHarrisFeatures(input_img, blockSize=5, ksize=3, k=0.04, threshold=0.1):
    # Returns detected Harris corner locations
    # Reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    cornerness = cv.cornerHarris(input_img, blockSize=blockSize,
                                 ksize=ksize, k=k)
    # threshold cornerness to receive corner locations
    corners = np.argwhere(cornerness > threshold*cornerness.max())
    return corners

def insertMarker(input_img, centers, marker_size=5):
    # Marks given coordinates on input image
    for i in range(0, len(centers)):
        input_img = cv.circle(input_img, [centers[i][1],centers[i][0]], marker_size, (0, 255, 0))
    return input_img

def convertToKeypoints(corners, size=13):
    # Converts a list of locations to cv Keypoints()
    corners = corners.astype(float)
    kps = []
    for i in range(0,len(corners)):
        kps.append(cv.KeyPoint(corners[i][1],corners[i][0], size))

    kps = np.asarray(kps)
    return kps

def getPointLists(feature_pairs, source_points, destination_points):
    # Extracts points locations from cv feature pairs
    src_features = []
    dst_features = []
    for i in range(0,len(feature_pairs)):
        src_features.append(source_points[feature_pairs[i].queryIdx].pt)
        dst_features.append(destination_points[feature_pairs[i].trainIdx].pt)

    return np.asarray(src_features).astype(float), np.asarray(dst_features).astype(float)

def stitchImages(trafos, images, img_size):
    # Stitches together the input images based on the given transforms.
    full_img = np.zeros(img_size, np.uint8)
    warps = []

    # warp the partial images into the full image
    for i in range(len(trafos)):
        warps.append(cv.warpAffine(images[i], trafos[i], (img_size[0], img_size[1])))
    # overlay the images

    full_img = cv.addWeighted(warps[0],0.6,warps[1],1.0,0)
    return full_img