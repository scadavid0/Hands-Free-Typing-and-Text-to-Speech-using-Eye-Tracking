There are two folders with source code, one for the gaze tracking code, and the other for
the virtual keyboard and the ball game. We describe their contents below:

GazeTracking code files:

__init__.py lets us use the TgtTracking class in the example.py, which is outside the 
main code folder.

mediapipe_gaze.py determines pupil positions in the camera frame using Google's mediapipe
library. 

OpenCVWebcam.py provides a wrapper for the openCV webcam, which we use, for example, when
debugging the gaze tracking code on a PC or mac instead of the RPI. 

PiCam.py provides a wrapper for the RPI Picam, which is used when we run the code on the RPI. 

target_tracking.py provides the main target tracking logic, so it has functionality to
do the calibration (i.e., learn a transformation) and then also to apply the transformation 
to a given pupil positions pair. transformation.py has methods for different transformations,
including a base class, affine transformation, and perspective transformation.

example.py contains the main program loop to run eye tracking and control cursor movement
with pyautogui.


VKeyboard-TTS code files:

Game_class.py implements a simple ball-tracking game using Pygame, where the user collects
balls with gaze tracking.

Keyboard_main.py implements an adjustable virtual keyboard designed for hands-free usage,
and is the main program to be run along with example.py in the GazeTracking folder.

Tune_function.py applies an EMA filter to the cursor input data to smooth movement and
reduce noise.