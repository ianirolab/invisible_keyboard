# A mix between finger tester and finger picker, improves image picking by guessing the finger 


import os
import tensorflow as tf
import cv2, requests
import numpy as np
import mediapipe as mp
import pickle
from in_model_manager import *


model = getTempModel()
# model = getModel1()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lastid = 0
notpushing = 0
cv2.startWindowThread()
picid = 0
if 'x2' not in os.listdir():  
    os.mkdir('./x2')
    os.mkdir('./y2')
if 'x-test2' not in os.listdir():  
    os.mkdir('./x-test2')
    os.mkdir('./y-test2')
elif len(os.listdir('./x')) > 0:
    a = [int(x) for x in os.listdir('./x')]
    picid = max(a) + 1
    # a = [int(x) for x in os.listdir('./x-test')]
    # picid = max(a) + 1
    print(picid)

# picid = 0

# cap = cv2.VideoCapture('./videos/v-0-70-6.mp4')
cap = cv2.VideoCapture('./videos/tester3.mp4')   
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
        # image = cv2.imread('./pictures/'+f+'.png')
        # with open('./results/'+f,'rb') as fl:
        #     result = pickle.load(fl)
        
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
        cv2.imshow('MediaPipe Hands', cv2.resize(cv2.flip(image, 1),(1280,720)))

        if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
            cv2.waitKey(1)
            continue
        
        x = results.multi_hand_landmarks
        # x = result
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
        # if model(np.array([tmp])) < 0.3:
        #     text = 'Raised'
        # else:
        #     text = 'Pushing'
        res = model(np.array([tmp])).numpy()
        rl = res.tolist()[0]
        idx = rl.index(max(rl))
        stautuses = ('sx_mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
        text = stautuses[idx]

        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow('MediaPipe Hands', image)

        
        
        k = cv2.waitKey(0)
        if (k==13):
            pass
        elif (k == ord('q')):
            picid -= 1
        elif (k == ord('a')):
            idx = 0
        elif (k == ord('s')):
            idx = 1
        elif (k == ord('d')):
            idx = 2
        elif (k == ord('f')):
            idx = 3
        elif (k == ord('c') or k == ord('v')):
            idx = 4
        elif (k == ord('j')):
            idx = 5
        elif (k == ord('k')):
            idx = 6
        elif (k == ord('l')):
            idx = 7
        elif (k == ord(';')):
            idx = 8
        # else:
        #     continue
        elif (k == ord('2')):
            continue
        else:
            idx = 9

        y = [0 for i in range(10)]
        y[idx] = 1
        # with open('./x2/'+str(picid),'wb') as f:
        #     pickle.dump(tmp,f)
        # with open('./y2/'+str(picid),'wb') as f:
        #     pickle.dump(y,f)
        # with open('./x-test2/'+str(picid),'wb') as f:
        #     pickle.dump(tmp,f)
        # with open('./y-test2/'+str(picid),'wb') as f:
        #     pickle.dump(y,f)



        picid+=1