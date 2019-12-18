import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    #plt.pause(0.0001)
    #ln[0].remove()
#-------------------------------------------------------------------------------
def sigmoid(score):
    # Formula of the sigmoid activation function
    return 1 / (1 + np.exp(-score))
#-------------------------------------------------------------------------------
# Calculates the error in its classifications, how well the line separates the
# data
def calculate_error(line_parameters, points, y):
    # Total number of points
    m = points.shape[0]
    # The probability of each point
    p = sigmoid(points * line_parameters)
    # general fromula for cross-entropy
    # -[SUM (yln(p) + (1-y)(ln(1-p))]
    cross_entropy = - (1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy
#-------------------------------------------------------------------------------
def gradient_descent(line_parameters, points, y, alpha):
    # Total number of points
    m = points.shape[0]
    for i in range(500):
        # The probability of each point
        p = sigmoid(points * line_parameters)
        # alpha: learning rate ensures that we are going to the direction
        # that best minimizes the line in small steps
        gradient = (points.T * (p - y)) * (alpha / m)
        # Results in new updates weight for a new line smaller error function
        # than the previous
        line_parameters -= gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b  = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = (-b - (w1 * x1)) / w2
    draw(x1, x2)
        #print(calculate_error(line_parameters, points, y))
#-------------------------------------------------------------------------------

# Number of total points in the model
n_pts = 500
bias = np.ones(n_pts)

# To get the same points each time the program is run
np.random.seed(0)

# Red points in the top right corner
top_region =  np.array([np.random.normal(10, 2, n_pts),
                        np.random.normal(12, 2, n_pts), bias]).T
# Blue points in the bottom left corner
bottom_region = np.array([np.random.normal(5, 2, n_pts),
                          np.random.normal(6, 2, n_pts), bias]).T

# Matrix (200, 3) containing all the coordinates as well as the bias
all_points = np.vstack((top_region, bottom_region))

# The wights and the bias
# w1x1 + w2x2 + b = 0
#w1 = -0.2
#w2 = -0.35
#b = 3.5

# Matrix (3, 1) containing the the weights and the bias
# Matrix : a 2D array with rows and cols
line_parameters = np.matrix([np.zeros(3)]).T

# The x-value of the left-most and the right-most points in the model graph
#x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])

# w1x1 + w2x2 + b = 0
#x2 = (-b - (w1 * x1)) / w2

""" (200, 1) Matrix which contains the linear combination of the points
which is the score of each points (Score determines if the the point
belongs to the red region or the blue region)"""
linear_combination = all_points * line_parameters

# Probability Matrix of (200, 1)
""" Based oon the points calculated in the linear_combination, using the sigmoid
function we obtain the probabily of a point being in the positive/negative
region"""
probabilities = sigmoid(linear_combination)

# Matrix of (200, 1) including of 100 zeros and 100 ones for the red and blue points
y = np.array([(np.zeros(n_pts), np.ones(n_pts))]).reshape(n_pts * 2, 1)

# Allows you to display multiple plots on the same figure
# it returns a tuple
_, ax = plt.subplots(figsize = (6, 6))
# coordinates of the points in the Top and Bottom
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'lightcoral')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'lightblue')

#draw(x1,x2)
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()

# Calculates the error of the line
#print(calculate_error(line_parameters, all_points, y))



# Comments

# The error can be written as: -[ln(P(red)) + ln(P(blue))]
# general fromula for cross-entropy
# -SUM (yln(p) + (1-y)(ln(1-p))

# Gradient Descent
# pts = points
# p = probability
# y = label
# m = number of points
#  _
# \/ E = (pts * (p - y)) / m
