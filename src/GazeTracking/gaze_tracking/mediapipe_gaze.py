# mediapipe_gaze.py:
 # This is the class for gaze tracking (from frame to pupil position) based on mediapipe
 # __init__() will initialize the mediapipe face_mesh engin and all the variables
 # getLandmarks(self,frame,is_BGR=True) will run the inference of mediapipe model and get 
 # the landmarks from camera frame
 # refresh(self,frame,is_BGR=True) will be called every frame, and it will call the 
 # getLandmarks(self,frame,is_BGR=True) and do some post-proccessing to the pupil position
 # pupil_left_coords(self) and pupil_right_coords(self) are utililty functions to 
 # calculate the pupil position on the camera frame (reverse of post-proccessing in refresh(self,frame,is_BGR=True))
 # annotated_frame(self) will add annotation to point out pupil positions on the camera frame

import cv2
import mediapipe as mp
import time

class mediapipe_gaze(object):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.eye_left_pupil_y  = None
        self.eye_left_pupil_x  = None
        self.eye_right_pupil_y = None
        self.eye_right_pupil_x = None
        
        self.eye_left_orig_x   = None
        self.eye_left_orig_y   = None
        self.eye_right_orig_x  = None
        self.eye_right_orig_y  = None

        self.pupils_located = False

        self.landmarks = None
        self.frame     = None

        self.frame_width  = None
        self.frame_height = None

    def getLandmarks(self,frame,is_BGR=True):
        if is_BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame.flags.writeable = False
        t = time.time()
        results = self.face_mesh.process(frame)
        print("process facemesh: ", (time.time()-t)*10**3, "ms")
        # frame.flags.writeable = True
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        else:
            return None
        
    def refresh(self,frame,is_BGR=True):
        self.frame_height, self.frame_width, _ = frame.shape
        t = time.time()
        if not is_BGR:
            results = self.getLandmarks(frame,is_BGR=False)
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            results = self.getLandmarks(frame)
            self.frame = frame
        print("get landmarks: ", (time.time()-t)*10**3, "ms")

        self.pupils_located = False
        t = time.time()
        if results:
            self.pupils_located = True
            self.landmarks = results.landmark

            self.eye_left_orig_y   = (self.landmarks[263].y + self.landmarks[362].y) / 2
            self.eye_left_orig_x   = (self.landmarks[263].x + self.landmarks[362].x) / 2
            self.eye_right_orig_y  = (self.landmarks[33].y  + self.landmarks[133].y) / 2
            self.eye_right_orig_x  = (self.landmarks[33].x  + self.landmarks[133].x) / 2

            self.eye_left_pupil_y  = self.landmarks[473].y - self.eye_left_orig_y
            self.eye_left_pupil_x  = self.landmarks[473].x - self.eye_left_orig_x 
            self.eye_right_pupil_y = self.landmarks[468].y - self.eye_right_orig_y
            self.eye_right_pupil_x = self.landmarks[468].x - self.eye_right_orig_x

            self.eye_left_orig_y   = int(self.frame_height * self.eye_left_orig_y  )
            self.eye_left_orig_x   = int(self.frame_width  * self.eye_left_orig_x  )
            self.eye_right_orig_y  = int(self.frame_height * self.eye_right_orig_y )
            self.eye_right_orig_x  = int(self.frame_width  * self.eye_right_orig_x )

            self.eye_left_pupil_y  = int(self.frame_height * self.eye_left_pupil_y )
            self.eye_left_pupil_x  = int(self.frame_width  * self.eye_left_pupil_x )
            self.eye_right_pupil_y = int(self.frame_height * self.eye_right_pupil_y)
            self.eye_right_pupil_x = int(self.frame_width  * self.eye_right_pupil_x)
        print("set gaze vars: ", (time.time()-t)*10**3, "ms")

    def pupil_left_coords(self):
        return self.eye_left_orig_x + self.eye_left_pupil_x, self.eye_left_orig_y + self.eye_left_pupil_y
    
    def pupil_right_coords(self):
        return self.eye_right_orig_x + self.eye_right_pupil_x, self.eye_right_orig_y + self.eye_right_pupil_y

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
