import tensorflow as tf
import cv2, requests
import numpy as np
import mediapipe as mp
import pickle
from in_model_manager import getModel0, getModel1

model = getModel1()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lastid = 0
notpushing = 0
cv2.startWindowThread()

cap = cv2.VideoCapture('./videos/tester2.mp4')
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while True:      
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
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.resize(cv2.flip(image, 1),(1280,720)))

        if (not results.multi_hand_landmarks )or len(results.multi_hand_landmarks) != 2:
            cv2.waitKey(1)
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
        # if model(np.array([tmp])) < 0.3:
        #     text = 'Raised'
        # else:
        #     text = 'Pushing'
        res = model(np.array([tmp])).numpy()
        rl = np.round(res,decimals=2).tolist()[0]
        
        stautuses = ('sx mig','sx_anu','sx_mid','sx_ind','thumb','dx_ind','dx_mid','dx_anu','dx_mig','raised')
        text = stautuses[rl.index(max(rl))]

        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow('MediaPipe Hands', image)

        # cv2.imwrite('./supposed_pushing/'+str(lastid)+'.png',image)
        # lastid += 1 
        # if model(np.array([tmp])).numpy()[0][0] > 0.8:
        #     cv2.imwrite('./supposed_pushing/'+str(lastid)+'.png',image)
        #     lastid += 1
        # else:
        #     notpushing += 1
        # print(notpushing)
        
        if cv2.waitKey(3) & 0xFF == 27:
            break