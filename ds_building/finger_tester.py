# Display live the results of the last neural networks trained for finger recognition
# the input frame are taken from input/pictures or webcam

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp
from in_model_manager import getFingerModel, subf
from globals import font,fontColor,fontScale,bottomLeftCornerOfText,thickness,lineType

model = getFingerModel()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lastid = 0
idx = 0
cv2.startWindowThread()

# pics = os.listdir('./ds_building/inputs/pictures')
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while True:      
        # res,image = cv2.imread('./ds_building/inputs/picutres/'+pics[idx])
        res,image = cap.read()
        if not res:
          break
        
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
        

        text = ''
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            
        
        
            x = results.multi_hand_landmarks
            tmp = []
            for i in range(21):

                tmp.append(x[0].landmark[i].x)
                tmp.append(x[0].landmark[i].y)
                tmp.append(x[0].landmark[i].z)
                
                tmp.append(x[1].landmark[i].x)
                tmp.append(x[1].landmark[i].y)
                tmp.append(x[1].landmark[i].z)

            
            res = model(np.array([tmp])).numpy()
            rl = np.round(res,decimals=2).tolist()[0]
            
            text = subf[rl.index(max(rl))]

        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow('Finger Tester', image)
        idx+=1

        cv2.waitKey(1)