from GUI import GUI
from HAL import HAL

import numpy as np
import cv2
import random

class Agent:

    def __init__(self):
        self.last_error = None
        self.focuspoint = None

    '''Artificial Vision Methods'''
    def color_segmentation(self, image, lower=[0, 0, 0], upper=[255, 0, 255]):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)

        return cv2.bitwise_and(image, image, mask = mask)

    def draw_point(self, image, point, radius=5, color=[255, 255, 255]):
        res = np.copy(image)
        res[point[0] - radius:point[0] + radius,
            point[1] - radius:point[1] + radius] = color
        
        return res

    def linecenter_at(self, image, h):
        original_img = np.copy(image)
        image = self.color_segmentation(image)

        kernel = np.ones((30,30), np.uint8)
        laplacian = cv2.erode(image, kernel, iterations = 1)
        laplacian = cv2.Laplacian(laplacian, cv2.CV_64F)

        index = np.argwhere(image[h, :, :] > 0)
        if index.size == 0:
            return None, original_img
        
        left = index[0, 0]
        right = index[-1, 0]
        center = [h, (left + right) // 2]

        return center, self.draw_point(original_img, center, color=[0, 0, 255])

    def get_linecenter(self, image, h):
        h1 = h - 15
        h2 = h1 + 200

        linecenter, image = self.linecenter_at(image, h)
        p1, image = self.linecenter_at(image, h1)
        p2, image = self.linecenter_at(image, h2)

        if any(x is None for x in [linecenter, p1, p2]):
            curve_type = 4
        elif abs(linecenter[1] - p1[1]) < 3 or abs(linecenter[1] - p2[1]) < 3:
            curve_type = 0
        elif abs(linecenter[1] - p1[1]) < 7 or abs(linecenter[1] - p2[1]) < 7:
            curve_type = 1
        elif abs(linecenter[1] - p1[1]) < 10 or abs(linecenter[1] - p2[1]) < 10:
            curve_type = 2
        else:
            curve_type = 3
        
        return linecenter, curve_type, image # Return image for debugging

    '''Controller Methods'''
    def error(self, image):
        h = (image.shape[0] // 16) * 9
        
        image = self.draw_point(image, self.focuspoint, color=[0, 255, 0])

        linecenter, is_rectline, image = self.get_linecenter(image, self.focuspoint[0])
        if linecenter:
            self.last_error = float(linecenter[1] - self.focuspoint[1]) # Normalize the error
        elif not self.last_error:
            return -1, is_rectline, image # Search the center of the line
        
        return self.last_error, is_rectline, image # Return image for debugging

last_e = 0
agent = Agent()

while True:
    image = HAL.getImage()

    ########### MY CODE ###########

    # [1] - Calculate the error
    e, curve_type, image = agent.error(image) # Image with interest points
    de = e - last_e
    last_e = e

    # [2] - Set hyperparameters
    if curve_type == 0:
        kp = 0.009
        kd = 0.01
        kv = 0.06
    elif curve_type == 1:
        kp = 0.009
        kd = 0.01
        kv = 0.045
    elif curve_type == 2:
        kp = 0.009
        kd = 0.01
        kv = 0.04
    elif curve_type == 3:
        kp = 0.009
        kd = 0.0125
        kv = 0.03
    else:
        kp = 0.009
        kd = 0.0125
        kv = 0.02

    # [3] - Calculate output of proportional controller (angular velocity)
    w = -(kp * e) - (kd * de)

    HAL.motors.sendW(w)

    # [4] - Claculate vertical velocity proportional to error
    maxw = image.shape[1] // 2
    v = kv * (maxw - abs(w))

    HAL.motors.sendV(v)

    print 'Error = ' + str(e)
    print 'Curve type = ' + str(curve_type)
    print '-(kp * e) = ' + str(-(kp * e))
    print '-(kd * de) = ' + str(-(kd * de))
    print 'V = ' + str(v)
    print 'W = ' + str(w)

    ###############################

    GUI.set_threshold_image(image)