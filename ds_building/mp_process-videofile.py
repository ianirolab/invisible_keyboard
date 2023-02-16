# gather frames from video file, process them with mp_hands and save image file and data file
# 
# This solution, although rather simple is pretty good since the camera can be moved anywhere
# but sometimes, especially in the more critical hand positions, mediapipe loses track of the hand
# and frames are lost until it tracks both hands again


import cv2
import mediapipe as mp
import pickle
import os
from in_model_manager import db_setup

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


idx = 0

db_setup('pictures','results')

cv2.startWindowThread()
for v in os.listdir('./ds_building/inputs/videos'):
  cap = cv2.VideoCapture('./ds_building/inputs/videos/'+v)
  length = str(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame = 0
  with mp_hands.Hands(
      model_complexity=1,
      min_detection_confidence=0.8,
      min_tracking_confidence=0.5) as hands:
    while True:
        
        res,image = cap.read()
        if not res:
          break
        
        frame += 1

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
        
        ######################################################################
        # Uncomment this section to watch mediapipe processing the video

        # img = cv2.flip(image, 1)
        # 
        # font                   = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (10,30)
        # fontScale              = 1
        # fontColor              = (0,255,0)
        # thickness              = 2
        # lineType               = 2
        # 
        # cv2.putText(img,str(frame) + '/' + length , 
        #     bottomLeftCornerOfText, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     thickness,
        #     lineType)
        # 
        # cv2.imshow("MediaPipe Process Video", cv2.resize(img,(1280,720)))
        ######################################################################

        print(str(frame) + '/' + length + '    ', end = '\r') 

        
        cv2.imwrite('./ds_building/inputs/pictures/'+str(idx)+'.png', cv2.flip(image,2))
        with open('./ds_building/inputs/results/' +str(idx),'wb') as f:
          pickle.dump(results.multi_hand_landmarks,f)

        idx += 1

        if cv2.waitKey(5) & 0xFF == 27:
          break