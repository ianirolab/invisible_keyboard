# gather frames from video file, process them with mp_hands and save image file and data file

import cv2
import mediapipe as mp
import numpy as np
import requests
import pickle
import os
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


idx = 0
commands = ['Push', 'Dont push'] 
command_timeout = 3
cidx = 1
lasttime = time.time()

if 'pictures' not in os.listdir(): 
    os.mkdir('./pictures')
    os.mkdir('./results')

cv2.startWindowThread()
for v in os.listdir('videos'):
  cap = cv2.VideoCapture('./videos/'+v)
  length = str(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame = 0
  with mp_hands.Hands(
      model_complexity=1,
      min_detection_confidence=0.8,
      min_tracking_confidence=0.8) as hands:
    while True:
        
        res,image = cap.read()
        if not res:
          break
        
        frame += 1
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
        
        img = cv2.flip(image, 1)
        # Write some Text

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,30)
        fontScale              = 1
        fontColor              = (0,255,0)
        thickness              = 2
        lineType               = 2

        cv2.putText(img,str(frame) + '/' + length , 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow("Output", cv2.resize(img,(1280,720)))

        if (time.time() > lasttime + command_timeout):
          lasttime = time.time()
          cidx = (cidx + 1) % len(commands)

        print(idx)
        if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
          cv2.waitKey(1)
          continue

        
        cv2.imwrite('./pictures/'+str(idx)+'.png', cv2.flip(image,2))
        with open('./results/' +str(idx),'wb') as f:
          pickle.dump(results.multi_hand_landmarks,f)

        idx += 1
        if (idx == 1000):
          break

        if cv2.waitKey(5) & 0xFF == 27:
          break