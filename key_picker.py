# View all recorded frames and classify them. This will create the dataset for the neural network
# To classify a frame, user will input a key, corresponding to the one that is "pressed" in the frame
# valid keys are all letters keys and ;,./
# pressing other keys will skip the frame

import os
import cv2
import pickle
from in_model_manager import key_db_setup

xid = 0

# xdir = 'x1'
# ydir = 'y1'
xdir = 'x-test1'
ydir = 'y-test1'

key_db_setup(xdir, ydir)

for dir in os.listdir('./inputs/results'):
    if (dir in ['raised','thumb']):
    # if (dir not in ['']):
        continue
    print(dir)
 
    fls = [int(fl) for fl in os.listdir('./inputs/results/'+dir+'/')]
    if (len(fls) == 0):
        continue

    # loop through pictures
    for i in range(max(fls)):
        if i not in fls:
            continue
        
        fl = str(i)
        image = cv2.imread('./inputs/pictures/'+dir+'/'+fl+'.png')
        
        cv2.imshow('Prediction',image)

        with open('./inputs/results/'+dir+'/'+fl,'rb') as f:
            result = pickle.load(f)
        

        
        k = cv2.waitKey(0)

        if (k >= 97 and k <= 122) or k in (ord(';'), ord(','), ord('.'), ord('/')):
            y = chr(k)
            with open('./inputs/'+xdir+'/'+dir+'/'+str(xid),'wb') as f:
                pickle.dump(result,f)
            with open('./inputs/'+ydir+'/'+dir+'/'+str(xid),'wb') as f:
                pickle.dump(y,f)
            xid+=1

        print(xid)