from GUI import GUI
from HAL import HAL

import math
import cv2
import numpy as np
import random

class Agent():
    global HAL, GUI, np, cv2, math, random

    def __init__(self):
        self.last_herror = 0
        self.last_verror = 0

    '''Artificial Vision Methods'''
    def color_segmentation(self, image, lower=[0, 0, 169], upper=[66, 255, 255]):
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(image, lower, upper)

        return cv2.bitwise_and(image, image, mask = mask)

    def draw_point(self, image, point, radius=5, color=[255, 255, 255]):
        res = np.copy(image)
        res[point[0] - radius:point[0] + radius,
            point[1] - radius:point[1] + radius] = color
        
        return res

    def get_linecenter(self, image, h):
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

        return center, self.draw_point(original_img, center, color=[255, 0, 0])

    '''Controller Methods'''
    def herror(self, image, focuspoint):
        image = self.draw_point(image, focuspoint, color=[0, 255, 0])

        linecenter, image = self.get_linecenter(image, focuspoint[0])

        if linecenter:
            self.last_herror = float(linecenter[1] - focuspoint[1])
        return self.last_herror, image

    def verror(self, image, focuspoint, h_high, h_low):
        linecenter, image = self.get_linecenter(image, focuspoint[0])
        lc_high, image = self.get_linecenter(image, focuspoint[0] - h_high)
        lc_low, image = self.get_linecenter(image, focuspoint[0] + h_low)

        if linecenter and lc_high and lc_low:
            self.last_verror = abs(float(linecenter[1] - lc_high[1])) + abs(float(linecenter[1] - lc_low[1]))
        return self.last_verror, image

    def error(self, image, focuspoint, h_high, h_low):
        _, image = self.herror(image, focuspoint)
        _, image = self.verror(image, focuspoint, h_high, h_low)
        return self.last_herror, self.last_verror, image

class Driver():
    global Agent, HAL, GUI, np, cv2, math, random

    def __init__(self):
        self.agent = Agent()
        self.last_herror = 0
        self.last_verror = 0
        self.iherror = 0

    def algorithm(self):
        while True:
            image = HAL.getImage()
    
            ########### MY CODE ###########
        
            # [1] - Set hyperparameters
            focuspoint = [(image.shape[0] // 16) * 9, 319]
            h_high, h_low = 22, 120
        
            is_line = self.last_verror < 7
            if is_line:
                maxv, kp, kd, kv, ki = 5, 0.0035, 0.007, 0.01, 0.000000905
            else:
                maxv, kp, kd, kv, ki = 3, 0.0035, 0.007, 0.007, 0.00000005
        
            # [2] - Calculate the error
            herror, verror, image = self.agent.error(image, focuspoint, h_high, h_low) # Image with interest points
            
            dherror = herror - self.last_herror
            dverror = verror - self.last_verror
            
            self.last_herror = herror
            self.last_verror = verror
            self.iherror += herror
        
            # [3] - Calculate velocity and angular velocity
            w = -(kp * herror) - (kd * dherror) - (ki * self.iherror)
            v = maxv -  (kv * verror)
        
            # [4] - Apply the changes
            HAL.motors.sendW(w)
            HAL.motors.sendV(v)
        
            # [5] - Log results 
            print('') 
            print('##### Errors #####')
            print('Horizontal error = ' + str(herror))
            print('Vertical errors error = ' + str(verror))
            print('##### Velocities #####')
            print('Horizontal velocity = ' + str(w))
            print('Vertical velocity = ' + str(v))
            print('Intregral error = ' + str(- (ki * self.iherror)))
        
            ###############################
        
            GUI.showImage(image)
            
Driver().algorithm()