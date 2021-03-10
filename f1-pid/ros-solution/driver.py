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

        ########### MY CODE ###########

        self.agent = Agent()
        self.last_herror = 0
        self.last_verror = 0
        self.iherror = 0

        ###############################

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

        # [1] - Set hyperparameters
        focuspoint = [(image.shape[0] // 16) * 9, 319]
        h_high, h_low = 22, 120

        is_line = self.last_verror < 10
        if is_line:
            maxv, kp, kd, kv, ki = 22.5, 0.01, 0.02, 0.04, 0.000017
        else:
            maxv, kp, kd, kv, ki = 13, 0.01, 0.015, 0.04, 0.000013

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
        self.motors.sendW(w)
        self.motors.sendV(v)

        # [5] - Log results 
        print '' 
        print '##### Errors #####'
        print 'Horizontal error = ' + str(herror)
        print 'Vertical errors error = ' + str(verror)
        print '##### Velocities #####'
        print 'Horizontal velocity = ' + str(w)
        print 'Vertical velocity = ' + str(v)
        print 'Intregral error = ' + str(- (ki * self.iherror))

        ###############################

        self.set_threshold_image(image)