import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import mediapipe as mp
import pickle
from in_model_manager import *


# model = getTempModel()
model = getModel1()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lastid = 0
notpushing = 0
cv2.startWindowThread()
fingers = []
picid = 0

# cap = cv2.VideoCapture('./history/videos/v-0-50-1.mp4')
cap = cv2.VideoCapture('./videos/tester2.mp4')   
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while True:      
        res,image = cap.read()
        if not res:
            break
        
    
        results = hands.process(image)

        
    

        if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
            continue
        
        x = results.multi_hand_landmarks
        tmp = []
        for i in range(21):

            tmp.append(x[0].landmark[i].x)
            tmp.append(x[0].landmark[i].y)
            tmp.append(x[0].landmark[i].z)
            
            tmp.append(x[1].landmark[i].x)
            tmp.append(x[1].landmark[i].y)
            tmp.append(x[1].landmark[i].z)

        
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,30)
        fontScale              = 1
        fontColor              = (0,255,0)
        thickness              = 2
        lineType               = 2

        res = model(np.array([tmp])).numpy()
        rl = res.tolist()[0]
        idx = rl.index(max(rl))
        fingers.append((idx,time.time()))

with open('fingerplot','wb') as f:
    pickle.dump(fingers,f)