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
def draw_lines(image, lines):
    # black image with the same size as the original image
    mask = np.zeros_like(image)
    #check if there is any detected lines
    if lines is not None:
        for line in lines:
            # The original detected line array is an array of 2D arrays
            #[[[569 282 990 703]] [[704 426 970 702]] ....]
            # the array is reshaped and unpacked to the coordinates of each
            #line given the x,y coordinate of the starting and ending points
            x1, y1, x2, y2 = line.reshape(4)
            # draw the lines into the mask image givien the coordinates
            cv2.line(mask, (x1, y1), (x2, y2), (100,100,100), 10)

    return mask
#-------------------------------------------------------------------------------
def merged_lines(image, lines):
    left_lines_list  = []
    right_lines_list = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        #slope = (y2 - y1) / (x2 - x1)
        #intercept = y1 - (slope * x1)
        if slope > 0:
            right_lines_list.append((slope, intercept))
        else:
            left_lines_list.append((slope, intercept))

    average_left_list  = np.average(left_lines_list, axis = 0)
    average_right_list = np.average(right_lines_list, axis = 0)

    left_line  = find_coordinates(image, average_left_list)
    right_line = find_coordinates(image, average_right_list)
    return np.array([left_line, right_line])
#-------------------------------------------------------------------------------
def find_coordinates(image, avg_lines):
    slope, intercept = avg_lines
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])
#-------------------------------------------------------------------------------

"""
# Loading the image
original_image = cv2.imread('test_image.jpg')

# Create a deep copy of the image matrix
duplicated_image = np.copy(original_image)

# transforms the image where only the gradient lines are visible
canny_image = canny(duplicated_image)

# crops the image where only the lanes are visible
cropped_image = lanes_region(canny_image)

#Since the in the x,y coordinate space the slope of a vertical line can't be
#calculated and it'll turn out to be infinity, we calculate the equation of the
#line in polar coordinate
#    Equation of the line in polar coordinate:
#        rho = xcos(theta) + ysin(theta)
detected_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
                                minLineLength = 40, maxLineGap= 100)

# average all the detected lines and merge all into one single lines
average_line = merged_lines(duplicated_image, detected_lines)

# draw the linee on a black image
line_image = draw_lines(duplicated_image, average_line)

#merged_image = np.bitwise_or(duplicated_image, line_image)
merged_image = cv2.addWeighted(duplicated_image, 0.8, line_image, 1, 1)

#plt can do the following work
cv2.imshow('Lanes', merged_image)
# Displays the image for the specifed amount of mili-second
cv2.waitKey(0)

#plt.imshow(lanes_region(canny_image))
#plt.show()
"""

cap = cv2.VideoCapture("test_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    # transforms the image where only the gradient lines are visible
    canny_image = canny(frame)

    # crops the image where only the lanes are visible
    cropped_image = lanes_region(canny_image)

    #Since the in the x,y coordinate space the slope of a vertical line can't be
    #calculated and it'll turn out to be infinity, we calculate the equation of the
    #line in polar coordinate
    #    Equation of the line in polar coordinate:
    #        rho = xcos(theta) + ysin(theta)
    detected_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
                                    minLineLength = 40, maxLineGap= 100)

    # average all the detected lines and merge all into one single lines
    average_line = merged_lines(frame, detected_lines)

    # draw the linee on a black image
    line_image = draw_lines(frame, average_line)

    #merged_image = np.bitwise_or(duplicated_image, line_image)
    merged_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("result", merged_image)
    cv2.waitKey(1)
