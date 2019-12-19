import cv2
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def canny(image):
    # Convert the image into graysclae
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blur the image to reduce noise / kernel of size 5,5 and deviation of 0
    # noise can create false edges and affect edge detection
    blur_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)

    # calculates the derivative in all direction of the adjacent pixels in the image
    # outline the strongest edges in the image
    canny_image = cv2.Canny(blur_image, 50, 150)

    return canny_image
#-------------------------------------------------------------------------------
def lanes_region(image):
    # Image Shape (704, 1279, 3)
    # height of the image
    height = image.shape[0]
    # put the the 3 (x, y) points where the triangle is being drawn
    triangle_region = np.array([ [
                           (200, height),
                           (1100, height),
                           (550, 250)
                        ] ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle_region, 255)
    # performs bitwise "and" operation of the canny image and the polygon mask
    # to crop the images and diplay the lane only
    outlined_image = cv2.bitwise_and(image, mask)
    return outlined_image
#-------------------------------------------------------------------------------
"""
Since the in the x,y coordinate space the slope of a vertical line can't be
calculated and it'll turn out to be infinity, we calculate the equation of the
line in polar coordinate
    Equation of the line in polar coordinate:
        rho = xcos(theta) + ysin(theta)
"""


#-------------------------------------------------------------------------------

# Loading the image
original_image = cv2.imread('test_image.jpg')

# Create a deep copy of the image matrix
duplicated_image = np.copy(original_image)

# transforms the image where only the gradient lines are visible
canny_image = canny(duplicated_image)

# crops the image where only the lanes are visible
cropped_image = lanes_region(canny_image)

#plt can do the following work
cv2.imshow('Lanes', cropped_image)
# Displays the image for the specifed amount of mili-second
cv2.waitKey(0)


#plt.imshow(lanes_region(canny_image))
#plt.show()
