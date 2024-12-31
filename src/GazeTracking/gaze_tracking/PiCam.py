# PiCam.py:
 # Wrapper for Raspberry Pi cam. will initialize the PI cam, and return camera frame when read was called
 # Used on Raspberry Pi for deployment

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput

class OpenCVWebcam(object):
    def __init__(self,width=640,height=480,fps=30):
        self.picam2 = Picamera2()
        self.video_config = self.picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"}, controls={"FrameRate": 30}) # Add fps setting?
        self.picam2.configure(self.video_config)
        self.encoder = H264Encoder(1000000, repeat=True)
        self.encoder.output = CircularOutput()
        self.picam2.start()
        self.picam2.start_encoder(self.encoder)

        (self.webcam_width, self.webcam_height) = self.picam2.video_configuration.size
        self.webcam_fps = 30 

    def getSettings(self): # No need to change
        return self.webcam_width, self.webcam_height, self.webcam_fps
    
    def read(self): # Return one frame
        return True, self.picam2.capture_array("main").reshape((self.webcam_height,self.webcam_width,3)).astype(np.uint8)

    
    def __del__(self):
        pass