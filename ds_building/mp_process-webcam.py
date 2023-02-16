# gather frames from webcam, process them with mp_hands and save image file and data file
# 
# This is the reccomended solution to gather data: watching MediaPipe live processing images
# avoid time wastes when MP loses track of hands. Plus it's possible to call an FingerModel
# to catch critical hand positions that model doesn't predict correctly.  
# All of this with many more frames and a better resolution than the mp_process_url.py solution


import cv2
import mediapipe as mp
import numpy as np
import pickle
from in_model_manager import getFingerModel, db_setup, subf


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


pidx = 0
model = getFingerModel()
db_setup


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
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

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
      
      
          
          res = model(np.array([tmp])).numpy()
          rl = res.tolist()[0]
          idx = rl.index(max(rl))
          text = subf[idx]

          
          
          
      image = cv2.flip(image, 1)
      cv2.putText(image,text, 
          bottomLeftCornerOfText, 
          font, 
          fontScale,
          fontColor,
          thickness,
          lineType)

      cv2.imshow('MediaPipe Process Webcam', image)

      if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        cv2.imwrite('./ds_building/input/pictures/'+str(pidx)+'.png', cv2.flip(image,2))
        with open('./ds_building/input/results/' +str(pidx),'wb') as f:
          pickle.dump(results.multi_hand_landmarks,f)

        pidx += 1

      print(pidx)

      if cv2.waitKey(1) & 0xFF == 27:
        break