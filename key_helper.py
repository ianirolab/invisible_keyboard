# Display live the results of the last neural networks trained for key recognition and save the frames.
# Having guesses while recording pictures helps finding hand position with which the neural network
# is not familiar with 

import cv2
import numpy as np
import mediapipe as mp
import pickle
from in_model_manager import  getFingerModel, getKeyModel, key_db_setup
from globals import key_map,font,fontColor,fontScale,bottomLeftCornerOfText,thickness,lineType

stautuses = tuple(key_map.keys())

# Load Models
key_models = {}
for k in key_map:
    key_models[k] = getKeyModel(k)

fingerModel = getFingerModel()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

key_db_setup('pictures', 'results')

cv2.startWindowThread()
cap = cv2.VideoCapture(0)   
# cap = cv2.VideoCapture('./videos/v-0-70-6.mp4')

picid = 0

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
    
    while True:      
        # Read frame
        res,image = cap.read()
        if not res:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(image)

        # Draw MediaPipe hands on the image.
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
        finger = -1

        # Process MediaPipe hand points only if both hands are showing
        if ( results.multi_hand_landmarks )and len(results.multi_hand_landmarks) == 2:
            
            # convert landmarks to python list
            x = results.multi_hand_landmarks
            tmp = []
            for i in range(21):

                tmp.append(x[0].landmark[i].x)
                tmp.append(x[0].landmark[i].y)
                tmp.append(x[0].landmark[i].z)
                
                tmp.append(x[1].landmark[i].x)
                tmp.append(x[1].landmark[i].y)
                tmp.append(x[1].landmark[i].z)

            # process points through FingerModel
            res = fingerModel(np.array([tmp])).numpy()
            rl = res.tolist()[0]
            finger = rl.index(max(rl))
            
            if finger == 9:
                text = 'raised'
            elif finger == 4:
                text = 'space'
            else:
                # Find left and right hands, to process only the hand related to the finger that is pushing
                if results.multi_handedness[0].classification[0].label == 'Left':
                    left = 1
                else:
                    left = 0
                
                right = 1-left
                rl = left

                if finger > 4:
                    finger -=1
                    rl = right
                tmp = []
                for i in range(21):

                    tmp.append(x[rl].landmark[i].x)
                    tmp.append(x[rl].landmark[i].y)
                    tmp.append(x[rl].landmark[i].z)
                
                # Calculate the result that the old models would give
                res = key_models[stautuses[finger]](np.array([tmp])).numpy()
                rl = res.tolist()[0]
                key = rl.index(max(rl))
                text = key_map[stautuses[finger]][key]
                
                # Save picture
                cv2.imwrite('./inputs/pictures/'+stautuses[finger]+'/'+str(picid)+'.png',cv2.flip(image,1))
                with open('./inputs/results/'+stautuses[finger]+'/'+str(picid),'wb') as fl:
                    pickle.dump(tmp,fl)

                picid+=1

        # Display image in selfie view with old models guesses
        image = cv2.resize(cv2.flip(image, 1),(1280,720))
        
        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(1)
        
        

        

