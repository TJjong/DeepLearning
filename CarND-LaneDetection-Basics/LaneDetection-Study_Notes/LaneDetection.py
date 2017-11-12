# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:13 2017

@author: HC
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# read in the image and print out some states
image = mpimg.imread("D:/STUDY/17-Python/LaneDetection/test.jpg")
print('This image is: ', type(image), 'with dimensions: ',image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

#######################
### Color Selection ###
#######################

# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

# Define color selection criteria
red_threshold = 210
green_threshold = 210
blue_threshold = 210
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold
thresholds =   (image[:,:,0] < rgb_threshold[0])\
              |(image[:,:,1] < rgb_threshold[1])\
              |(image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
plt.show()

# Change red_threshold, green_threshold and blue_threshold until enough desired imformation is retained.

######################
### Region Masking ###
######################

region_select = np.copy(image)
# Define a triangle region of interest
# Keep in mind the origin (x = 0, y = 0) is in the upper left in the image processing
left_bottom = [0,539]
right_bottom = [900,539]
apex = [400,300]

# Fit lines (y = Ax + B) to identify the 3 sided region of interest
# np.polyfit() returns the coefficients [A,B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0,xsize), np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) &\
                    (YY > (XX*fit_right[0] + fit_right[1])) &\
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                    
# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255,0,0]

plt.imshow(region_select)

# change left_bottom, right_bottom, apex to see different results

###########################################
### Combine Color and Region Selections ###
###########################################

color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 210
green_threshold = 210
blue_threshold = 210
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest (Note: if you run this code, 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [450, 300]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                    
# Mask color selection
color_select[color_thresholds] = [0,0,0]

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, 'b--', lw=4)

plt.imshow(color_select)
plt.imshow(line_image)

############################
### Canny Edge Detection ###
############################

# import an image
image = mpimg.imread("D:/STUDY/17-Python/LaneDetection/exit-ramp.jpg")
plt.imshow(image)

import cv2 # bringing in OpenCV libraries
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # grayscale conversion
plt.imshow(gray, cmap = 'gray')

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')

#####################################
### Hough Transform to Find Lines ###
#####################################

# read in and grayscale the image
image = mpimg.imread("D:/STUDY/17-Python/LaneDetection/exit-ramp.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
# define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# define parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# define the Hough transform parameters
# make a blank the same size as the image to draw on 
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 40
max_line_gap = 1
line_image = np.copy(image)*0 # create a blank to draw lines on

# run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

# draw lines on the blank 
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0),10)
plt.imshow(line_image)

# create a 'color' binary image to combine with line image 
color_edges = np.dstack((edges, edges, edges))

# draw the line on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1,  0)
plt.imshow(combo)


###################################
### Add Mask to Hough Transform ###
###################################

# read in and grayscale the image
image = mpimg.imread("D:/STUDY/17-Python/LaneDetection/exit-ramp.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
# define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# define parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# before implementing the Hough transform, we first create a maked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

image_shape = image.shape
vertices = np.array([[(0,image_shape[0]),(450,300),(500,300), (image_shape[1], image_shape[0])]], 
                      dtype = np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
#plt.imshow(masked_edges)

#X = [0,450,500,image_shape[1]]
#Y = [image_shape[0],300,300,image_shape[0]]
#plt.plot(X, Y, 'b--', lw=4)

# define the Hough transform parameters
# make a blank the same size as the image to draw on 
rho = 1              # distance resolution
theta = np.pi/180    # angular resolution
threshold = 15        # minimum number of votes
min_line_length = 40 # minimum number of pixels making up a line
max_line_gap = 20     # maximum gap in pixels between connectable lien segments
line_image = np.copy(image)*0 # create a blank to draw lines on

# run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

# draw lines on the blank 
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0),10)
plt.imshow(line_image)

# create a 'color' binary image to combine with line image 
color_edges = np.dstack((edges, edges, edges))

# draw the line on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1,  0)
plt.imshow(combo)










