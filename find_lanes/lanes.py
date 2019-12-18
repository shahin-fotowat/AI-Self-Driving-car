import cv2
import numpy as np

# Loading the image
original_image = cv2.imread('test_image.jpg')
# Create a deep copy of the image matrix
duplicated_image = np.copy(original_image)
#Display the image

# Convert the image into graysclae
grayscale_imagee = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# blur the image to reduce noise / kernel of size 5,5 and deviation of 0
blur_image = cv2.GaussianBlur(grayscale_imagee, (5,5), 0)

cv2.imshow('Lanes', blur_image)
# Displays the image for the specifed amount of mili-second
cv2.waitKey(0)

# noise can create false edges and affect edge detection
