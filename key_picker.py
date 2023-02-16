import os
import cv2
import pickle
from ds_building.in_model_manager import key_db_setup, key_db_setup_test

xid = 0

if 'x2' not in os.listdir():
    key_db_setup()
if 'x-test2' not in os.listdir():
    key_db_setup_test()

for dir in os.listdir('results'):
    # if (dir in ['raised','thumb','sx_ind','dx_ind','dx_anu','sx_anu','sx_mid']):
    if (dir not in ['sx_ind','dx_ind']):
        continue
    print(dir)
 
    fls = [int(fl) for fl in os.listdir('results/'+dir+'/')]
    for i in range(max(fls)):
        if i not in fls:
            continue
        
        fl = str(i)
        image = cv2.imread('pictures/'+dir+'/'+fl+'.png')
        cv2.imshow('Picture',image)
        with open('results/'+dir+'/'+fl,'rb') as f:
            result = pickle.load(f)
        

        k = cv2.waitKey(0)
        

        if (k >= 97 and k <= 122) or k in (ord(';'), ord(','), ord('.'), ord('/')):
            y = chr(k)
            # with open('./x2/'+dir+'/'+str(xid),'wb') as f:
            #     pickle.dump(result,f)
            # with open('./y2/'+dir+'/'+str(xid),'wb') as f:
            #     pickle.dump(y,f)
            with open('./x-test3/'+dir+'/'+str(xid),'wb') as f:
                pickle.dump(result,f)
            with open('./y-test3/'+dir+'/'+str(xid),'wb') as f:
                pickle.dump(y,f)
            xid+=1

        print(xid)