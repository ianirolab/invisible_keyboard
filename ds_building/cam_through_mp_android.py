# gather frames from url, process them with mp_hands and save image file and data file

import cv2
import mediapipe as mp
import numpy as np
import requests
import pickle
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

url = "http://localhost:5555/shot.jpg"

idx = 0

if 'pictures' not in os.listdir(): 
    os.mkdir('./pictures')
    os.mkdir('./results')

cv2.startWindowThread()
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
  while True:
      # print(idx)
      img_resp = requests.get(url)
      img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
      img = cv2.imdecode(img_arr, -1)
    
      image = img
      
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
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
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