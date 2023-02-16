# gather frames from video file, process them with mp_hands and save image file and data file

import cv2
import mediapipe as mp
import numpy as np
import requests
import pickle
import os
import time
from in_model_manager import *


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


pidx = 0
commands = ['Push', 'Dont push'] 
command_timeout = 3
cidx = 1
lasttime = time.time()
model = getTempModel()

if 'pictures' not in os.listdir(): 
    os.mkdir('./pictures')
    os.mkdir('./results')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2

cv2.startWindowThread()
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6) as hands:
  while True:
      text = ''
      res,image = cap.read()
      if not res:
        break
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

        x = results.multi_hand_landmarks
        if len(x) == 2:
          tmp = []
          for i in range(21):

              tmp.append(x[0].landmark[i].x)
              tmp.append(x[0].landmark[i].y)
              tmp.append(x[0].landmark[i].z)
              
              tmp.append(x[1].landmark[i].x)
              tmp.append(x[1].landmark[i].y)
              tmp.append(x[1].landmark[i].z)
      
      
          
          # if model(np.array([tmp])) < 0.3:
          #     text = 'Raised'
          # else:
          #     text = 'Pushing'
          res = model(np.array([tmp])).numpy()
          rl = res.tolist()[0]
          idx = rl.index(max(rl))
          stautuses = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
          text = stautuses[idx]

          
          
          
        # image = cv2.resize(cv2.flip(image, 1),(1280,720))
      image = cv2.flip(image, 1)
      cv2.putText(image,text, 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          thickness,
          lineType)

      cv2.imshow('MediaPipe Hands', image)

      # if (time.time() > lasttime + command_timeout):
      #   lasttime = time.time()
      #   cidx = (cidx + 1) % len(commands)

      # print(idx)
      # if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
      #   cv2.waitKey(1)
      #   continue

      if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        cv2.imwrite('./pictures/'+str(pidx)+'.png', cv2.flip(image,2))
        with open('./results/' +str(pidx),'wb') as f:
          pickle.dump(results.multi_hand_landmarks,f)

        pidx += 1

      print(pidx)

      if cv2.waitKey(1) & 0xFF == 27:
        break