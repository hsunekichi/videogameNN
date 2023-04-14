from inputs import get_gamepad
import math
import threading
import time
import sys
import cv2
import numpy as np
import pyautogui
import config as conf
import mss


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        left_stick_x = self.LeftJoystickX
        left_stick_y = self.LeftJoystickY

        x_btn = self.Y  # El mando de xbox tiene los botones X e Y invertidos 
        a_btn = self.A
        b_btn = self.B
        rt = self.RightTrigger
        
        return [left_stick_x, left_stick_y, 
                    x_btn, a_btn, b_btn, rt]
    
        #return [left_stick_x, left_stick_y, a_btn]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.X = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state

sct = mss.mss()

# Captures a frame
def getScreen():
    initX = 74
    initY = 27

    endX = initX + conf.screen_width
    endY = initY + conf.screen_height
    
    # Capture the screen
    # img = pyautogui.screenshot(region=(initX, initY, conf.screen_width, conf.screen_height))
    
    # The screen part to capture
    region = {'top': initY, 'left': initX, 'width': conf.screen_width, 'height': conf.screen_height}

    # Grab the data
    img = sct.grab(region)

    # Convert the image into numpy array
    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert the color space from BGRA to BGR
    # Write the frame into the file 'output.avi'

    #cv2.imshow('frame', img)
    #cv2.waitKey(1)

    return img 


def getInput(joy):
    return joy.read()

