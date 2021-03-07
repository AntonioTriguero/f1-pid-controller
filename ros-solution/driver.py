#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
from datetime import datetime

import math
import cv2
import numpy as np

from agent import Agent

time_cycle = 80

class Driver(threading.Thread):

    def __init__(self, camera, motors):
        self.camera = camera
        self.motors = motors
        self.threshold_image = np.zeros((640,360,3), np.uint8)
        self.color_image = np.zeros((640,360,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()

        self.agent = Agent()
        image = self.getImage()
        h = (image.shape[0] // 16) * 9
        self.agent.focuspoint = [h, 319]

        self.last_e = 0

        threading.Thread.__init__(self, args=self.stop_event)
    
    def getImage(self):
        self.lock.acquire()
        img = self.camera.getImage().data
        self.lock.release()
        return img

    def set_color_image (self, image):
        img  = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.color_image_lock.acquire()
        self.color_image = img
        self.color_image_lock.release()
        
    def get_color_image (self):
        self.color_image_lock.acquire()
        img = np.copy(self.color_image)
        self.color_image_lock.release()
        return img
        
    def set_threshold_image (self, image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
        
    def get_threshold_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.algorithm()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
        image = self.getImage()

        ########### MY CODE ###########

        # [1] - Calculate the error
        e, curve_type, image = self.agent.error(image) # Image with interest points
        de = e - self.last_e
        self.last_e = e

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

        self.motors.sendW(w)

        # [4] - Claculate vertical velocity proportional to error
        maxw = image.shape[1] // 2
        v = kv * (maxw - abs(w))

        self.motors.sendV(v)

        print 'Error = ' + str(e)
        print 'Curve type = ' + str(curve_type)
        print '-(kp * e) = ' + str(-(kp * e))
        print '-(kd * de) = ' + str(-(kd * de))
        print 'V = ' + str(v)
        print 'W = ' + str(w)

        ###############################

        self.set_threshold_image(image)