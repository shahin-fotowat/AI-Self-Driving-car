import cv2
import numpy as np
#import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))

+-+-    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
#-------------------------------------------------------------------------------
def average_sloped_intercept(image, lines):
    # will containt the coordinates of the average lines on the left
    left_fit = []
    # will containt the coordinates of the average lines on the right
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # we'll fit a first degree polynomial which will be a linear function
        # y = mx + b. it'll fit our polynomial to 'x' and 'y' points and return
        # a vector of coefficients which describes the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
    ``    else:
            right_fit.append((slope, intercept))

    left_fit_average  = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)

    left_line  = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
#-------------------------------------------------------------------------------
def canny(image):
    #------------------- Changing image color --------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #------------------- Reduce noise and smoothening ------------------
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #------------------- Identify edges in the image -------------------
    #it perfoems a derivative on our function in both 'x' and 'y' directions
    #thereby measuring the change in intesity with respect to adjancet pixels
    #a small derivative is a small change in intesity and a big derivative is a
    #big change
    #                 (image, low_threshold, high_treshold)
    canny = cv2.Canny(blur, 50, 150)

    return canny
#------------------------------------------------------------------------------
def display_lines(image, lines):
    """
    lines is a 3D array and  [[[704 418 927 641]]

                             [[704 426 791 516]]

                             [[320 703 445 494]]
                                    ...
                                             ]]]
    this is an example of the format. each row is a 2D array and here we
    want to convert every row from a 2D array to a 1D array
    [[x1, y1, x2, y2]]    ---------->   [x1, y1, x2, y2]
    """
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)

    return line_image
#------------------------------------------------------------------------------
def region_of_interest(image):
    #height of the image
    print("print image ", image.shape[0])
    height = image.shape[0]
    polygon = np.array([
                    [(200, height), (1100, height), (550, 250)]
                ])

    """creates an array of zeros with the same shape as the image's corresponding
    array, both arrays will have the same number of rows and cols
    same amount of pixels and dimention, they'll have 0 intensities"""
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    """
    Computing the bitwise '&' of both images, takes the bitwise '&' of each
    homologous pixel in both arrays, ultimately masking the canny image to only
    show the region of interest traced by the polygonal contour of the mask
    """
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#------------------------------------------------------------------------------

# image = cv2.imread('test_image.jpg')
#
# #create a copy of the pixels array. changes to lane_image won't affect
# #the image array
# lane_image = np.copy(image)
#
# #convert the color space of the image to another color which is gray here
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
#
# """
# Threshold: minimum number of votes needed to accept a candidate line = (100)
# minLineLength: any detected line traced by less than 40 pixels are rejected
# maxLineGap: indicates the maximum  distance in pixels between segmented lines
# which we will allow to be connected into a single line instead of them being
# broken up
# """
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
#                         minLineLength=40, maxLineGap=5)
# print("printing lines: ", lines)
# averaged_lines = average_sloped_intercept(lane_image, lines)
#
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result2', combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
                            minLineLength=40, maxLineGap=5)
    print("printing lines: ", lines)
    averaged_lines = average_sloped_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result2', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#plt.imshow(canny)
#plt.show()
