# A mix between finger tester and finger picker, improves image picking by guessing the finger 


import os
import tensorflow as tf
import cv2, requests
import numpy as np
import mediapipe as mp
import pickle
from ds_building.in_model_manager import *

key_map = {'sx_mig':['q','a','z'],'sx_anu':['w','s','x'],'sx_mid':['e','d','c'],'sx_ind':['r','t','f','g','c','v'],
          'dx_ind':['y','u','h','j','n','m'],'dx_mid':['i','k',','],'dx_anu':['o','l','.'],'dx_mig':['p',';','/']}

landmarks_idxs = {'sx_mig':[0,1,5,9,13,17,14,15,16,18,19,20],'sx_anu':[0,1,5,9,13,17,10,11,12,6,7,8,14,15,16,17,18,19,20],
                  'sx_mid':[0,1,5,9,13,17,6,7,8,10,11,12,14,15,16],'sx_ind':[0,1,5,9,13,17,2,3,4,6,7,8,10,11,12],
                  'dx_mig':[0,1,5,9,13,17,14,15,16,18,19,20],'dx_anu':[0,1,5,9,13,17,10,11,12,6,7,8,14,15,16,17,18,19,20],
                  'dx_mid':[0,1,5,9,13,17,6,7,8,10,11,12,14,15,16],'dx_ind':[0,1,5,9,13,17,2,3,4,6,7,8,10,11,12]}


stautuses = tuple(key_map.keys())
key_models = {}
for k in key_map:
    key_models[k] = getKeyModel(k)

model = getMidModel()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lastid = 0
notpushing = 0
cv2.startWindowThread()
picid = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2



# cap = cv2.VideoCapture('./videos/v-0-70-6.mp4')
# cap = cv2.VideoCapture('./videos/tester3.mp4')   
cap = cv2.VideoCapture(0)   
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.4) as hands:
    # for f in range(2390):
    #     f = str(f)
  while True:      
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
            # for i in range(len(results.multi_hand_landmarks)-1):
                mp_drawing.draw_landmarks(
                image,
                # results.multi_hand_landmarks[ind],
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.resize(cv2.flip(image, 1),(1280,720)))
        text = ''
        if ( results.multi_hand_landmarks )and len(results.multi_hand_landmarks) == 2:
            
            if results.multi_handedness[0].classification[0].label == 'Left':
                ind = 1
            else:
                ind = 0

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
            rl = res.tolist()[0]
            idx = rl.index(max(rl))
            if idx == 9:
                text = 'raised'
            elif idx == 4:
                text = 'space'
            else:
                if results.multi_handedness[0].classification[0].label == 'Left':
                    left = 1
                else:
                    left = 0
                    
                right = 1-left
                rl = left

                if idx > 4:
                    idx -=1
                    rl = right
                tmp = []
                for i in range(21):

                    tmp.append(x[rl].landmark[i].x)
                    tmp.append(x[rl].landmark[i].y)
                    tmp.append(x[rl].landmark[i].z)

                res = key_models[stautuses[idx]](np.array([tmp])).numpy()
                rl = res.tolist()[0]
                idx2 = rl.index(max(rl))

                text = stautuses[idx] + '/' + key_map[stautuses[idx]][idx2]
            


        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        # print(image,end='\r')
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(1)
        
        
        
        

