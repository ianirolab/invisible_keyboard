# View all recorded frames and classify them. This will create the dataset for the neural network
# To classify a frame, user will input a key, corresponding to the finger that is currently being pressed
# the key-finger relationship is the same as putting fingers on a keyboard in the standard position, 
# in the middle row. The only exception, for pratical reasons is the thumbs, which correspond to 
# 'c' or 'v' instead of spacebar, which is reserved for 'raised' status.
# Additionally it's also possible to undo last classification by pressing 2, and change the classification
# by pressing 'q'
#
# Results are stored in the file pushing.txt

import os
import cv2

l_mig = []
l_anu = []
l_mid = []
l_ind = []
thumb = []
r_mig = []
r_anu = []
r_mid = []
r_ind = []
raised = []

names = os.listdir('./ds_building/inputs/pictures')
names.sort()

cv2.namedWindow('Finger Picker')

i = 9000 
maxi = i + 1000
while (i < maxi ):
    try:
        fname = names[names.index(str(i) + '.png')]
        img = cv2.imread('./ds_building/inputspictures/' + fname)
    except:
        i+=1
        continue
    if(img is None):
        continue

    cv2.imshow(str(i), cv2.resize(img,(1280,720)))
    
    k = cv2.waitKey(0)
    if (k == ord('q')):
        i = i - 2
    elif (k == ord('a')):
        l_mig.append(fname)
    elif (k == ord('s')):
        l_anu.append(fname)
    elif (k == ord('d')):
        l_mid.append(fname)
    elif (k == ord('f')):
        l_ind.append(fname)
    elif (k == ord('c') or k == ord('c')):
        thumb.append(fname)
    elif (k == ord('j')):
        r_ind.append(fname)
    elif (k == ord('k')):
        r_mid.append(fname)
    elif (k == ord('l')):
        r_anu.append(fname)
    elif (k == ord(';')):
        r_mig.append(fname)
    elif (k == ord('2')):
        os.remove('./ds_building/inputs/pictures/' + fname)
        os.remove('./ds_building/inputs/results/' + fname[:-4])
    
    i += 1

    cv2.destroyAllWindows()


lists = [l_mig,l_anu,l_mid,l_ind,thumb,r_ind,r_mid,r_anu,r_mig, raised]

with open('./ds_building/pushing.txt','w') as f:
    for i in range(len(lists)):
        for img in lists[i]: 
            f.write(img +' '+str(i) +'\n') 