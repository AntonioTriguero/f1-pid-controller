import numpy as np
import cv2
import random

class Agent:

    def __init__(self):
        self.last_herror = 0
        self.last_verror = 0

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

        return center, self.draw_point(original_img, center, color=[0, 0, 255])

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