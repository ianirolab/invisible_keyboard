import tensorflow as tf
import cv2, requests
import numpy as np
import mediapipe as mp
import pickle


model = tf.keras.models.load_model('model_s_500-500-500-1001-20')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

url = "http://localhost:5555/shot.jpg"

lastid = 0
notpushing = 0
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
        text = str(model(np.array([tmp])).numpy()[0][0])
        image = cv2.flip(image,1)

        cv2.putText(image,text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.imshow('MediaPipe Hands', image)

        cv2.imwrite('./supposed_pushing/'+str(lastid)+'.png',image)
        lastid += 1 
        # if model(np.array([tmp])).numpy()[0][0] > 0.8:
        #   cv2.imwrite('./supposed_pushing/'+str(lastid)+'.png',image)
        #     lastid += 1
        # else:
        #     notpushing += 1
        # print(notpushing)
        
        if cv2.waitKey(1) & 0xFF == 27:
            # print(notpushing)
            break