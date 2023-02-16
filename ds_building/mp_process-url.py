# gather frames from url, process them with mp_hands and save image file and data file
# to setup the camera, make sure that refreshing http://.../shot.jpg returns the last 
# frame recorded by the camera
# 
# This solution, although the easiest one for live data gathering is not reccomended as the
# low framerate makes it hard for mediapipe to track hands and the quality of the dataset
# generated is generally lower

import cv2
import mediapipe as mp
import numpy as np
import requests
import pickle
from in_model_manager import db_setup

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

url = "http://localhost:5555/shot.jpg"

idx = 0

db_setup('pictures','results')

cv2.startWindowThread()
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while True:
      # print(idx)
      img_resp = requests.get(url)
      img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
      img = cv2.imdecode(img_arr, -1)
    
      image = img
      
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
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Process URL', cv2.flip(image, 1))
      print(idx)
      if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
        cv2.waitKey(1)
        continue

      
      cv2.imwrite('./ds_building/inputs/pictures/'+str(idx)+'.png', cv2.flip(image,2))
      with open('./ds_building/inputs/results/' +str(idx),'wb') as f:
        pickle.dump(results.multi_hand_landmarks,f)

      idx += 1

      if cv2.waitKey(5) & 0xFF == 27:
        break