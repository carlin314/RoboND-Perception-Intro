#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# prepare object points
nx = 8 #enter the number of inside corners in x
ny = 6 #enter the number of inside corners in y

# Make a list of calibration images
images = glob.glob('./cali_small/Cal*.jpg')

# Select any index to grab an image from the list
idx = 4

# Read in the image
img = mpimg.imread(images[idx])

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()
else:
    # Could not find corners, just show the original image
    print("\nCould not find corners!\n")
    plt.imshow(img)
    plt.show()
