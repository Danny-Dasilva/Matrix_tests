import numpy as np
# camera_matrix = np.array([[778.14666748,   0,         489.9304091, ]
                            # [  0,         781.06970215, 252.51055003,]
                             # [  0,           0,           1,        ]])

# dist_coeffs = [[ 0.11114798, -0.82947359, -0.00232797, -0.00172743,  0.54440471]]

#####3#

camera_matrix = np.array(
       [[  6.62201661e+03,   0.00000000e+00,   1.09300820e+03],
          [  0.00000000e+00,   7.39754133e+03,   6.68477741e+02],
          [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]],'float64')
dist_coeffs = np.array([  1.47718955e+00,  -1.32890661e+02,  -5.78854939e-02,
             1.54970815e-02,  -4.61317397e+00],'float64')


import math

import numpy as np
import scipy.optimize
import cv2
# Load captured image.
image_bgr = cv2.imread("box_location_input.png", cv2.IMREAD_COLOR)
# Convert to HSV color space.
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# Mask box
YELLOW = {"lower":(25, 100, 50),"upper":(35, 255, 200)}
GREEN = {"lower":(64, 96, 64),"upper": (90, 255, 200)}
mask = cv2.inRange(image_hsv, YELLOW["lower"],YELLOW["upper"])
mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
mask = cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)))
contours = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key = cv2.contourArea, reverse = True)
cv2.drawContours(image_bgr, contours, 0, (255, 255, 255), 3)
targetmask = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 1), np.uint8)
cv2.drawContours(targetmask, contours, 0, (255), -1)
cv2.imshow("img", targetmask)
cv2.waitKey(0)

# box_size = [0.150,0.100,0.075]
def boxcorners(box_size):
    # Define box corners in box coordinate system.
    half_size_x = box_size[0] / 2.0
    half_size_y = box_size[1] / 2.0
    half_size_z = box_size[2] / 2.0
    corners = np.array(
        [[-half_size_x, -half_size_y, -half_size_z],
         [+half_size_x, -half_size_y, -half_size_z],
         [-half_size_x, +half_size_y, -half_size_z],
         [+half_size_x, +half_size_y, -half_size_z],
         [-half_size_x, -half_size_y, +half_size_z],
         [+half_size_x, -half_size_y, +half_size_z],
         [-half_size_x, +half_size_y, +half_size_z],
         [+half_size_x, +half_size_y, +half_size_z]],
        'float64')
    return corners
def drawBoxOnImage(rotation_vector, translation_vector,camera_matrix,dist_coeffs, image):
        # Draw the box on a given (color) image, given the rotation and
        # translation vector.
        box_size = [0.150,0.100,0.075]
        corners = boxcorners(box_size)
        # Project box corners to image plane.
        pts = cv2.projectPoints(
            corners, rotation_vector, translation_vector,
            camera_matrix, dist_coeffs)[0]
        # Draw box on image
        projected_image = image.copy()
        cv2.polylines(
            projected_image,
            np.array([[pts[1][0], pts[0][0], pts[2][0], pts[3][0]],
                      [pts[0][0], pts[1][0], pts[5][0], pts[4][0]],
                      [pts[1][0], pts[3][0], pts[7][0], pts[5][0]],
                      [pts[3][0], pts[2][0], pts[6][0], pts[7][0]],
                      [pts[2][0], pts[0][0], pts[4][0], pts[6][0]],
                      [pts[4][0], pts[5][0], pts[7][0], pts[6][0]]], 'int32'),
            True, (0, 255, 0), 3)
        return projected_image
