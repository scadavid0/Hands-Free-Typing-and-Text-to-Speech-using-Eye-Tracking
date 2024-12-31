# example.py:
 # main loop to run eye tracking and control cursor movement. clamp() ensures the gaze
 # target coordinates remain with the bounds of the window. pyautogui moves the cursor to
 # the target (x,y) position. press 'q' to quit the calibration / program, or press 'i'
 # to recalibrate.

import cv2
from gaze_tracking.target_tracking import TgtTracking
import pyautogui

target = TgtTracking()
target.show_webcam(5)

# if not target.trans.init_calibration_done:
target.initial_calibrate()

# target.measure_error() # if you want to plot the error measures of our gaze tracking

def clamp(input_x,input_y):
    x = min(max(input_x,50),1870)
    y = min(max(input_y,50),1150)
    return x, y

while True:
    target.refresh(Display=False)

    x, y = clamp(target.target_x, target.target_y)
    pyautogui.moveTo(x, y, duration=0)

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('i'):
        print("Restart Calibration!")
        target.initial_calibrate()
