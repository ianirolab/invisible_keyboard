# Display live the results of the last neural networks trained for finger recognition and save the frames.
# Having guesses while recording pictures helps finding hand position with which the neural network
# is not familiar with 


import pickle
import cv2
import numpy as np
import mediapipe as mp
from in_model_manager import getFingerModel, db_setup, subf

xdir = 'x1'
ydir = 'y1'

db_setup('x1','y1')

model = getFingerModel()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2


cv2.startWindowThread()
cap = cv2.VideoCapture('./ds_building/inputs/videos/tester3.mp4')   

picid = 0
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.4) as hands:
  while True:      
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

        
        res = model(np.array([tmp])).numpy()
        rl = res.tolist()[0]
        idx = rl.index(max(rl))
        text = subf[idx]

        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow('Finger Helper', image)

        
        
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

        with open('./ds_building/inputs'+xdir+'/'+str(picid),'wb') as f:
            pickle.dump(tmp,f)
        with open('./ds_building/inputs'+ydir+'/'+str(picid),'wb') as f:
            pickle.dump(y,f)



        picid+=1