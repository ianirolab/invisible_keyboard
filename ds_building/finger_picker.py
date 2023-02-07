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

names = os.listdir('./pictures')
names.sort()

cv2.namedWindow('Picture')        # Create a named window
cv2.moveWindow('Picture', 0,0)

i = 976

while (i < len(names)):
    try:
        fname = names[names.index(str(i) + '.png')]
        img = cv2.imread('./pictures/' + fname)
    except:
        i+=1
        continue
    if(img is None):
        continue

    cv2.imshow('Picture', cv2.resize(img,(1280,720)))
    
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
    elif (k == ord('c')):
        thumb.append(fname)
    elif (k == ord('j')):
        r_ind.append(fname)
    elif (k == ord('k')):
        r_mid.append(fname)
    elif (k == ord('l')):
        r_anu.append(fname)
    elif (k == ord(';')):
        r_mig.append(fname)
    
    i += 1

    cv2.destroyAllWindows()
    if (k == ord('2')):
        os.remove('./pictures/' + fname)
        os.remove('./results/' + fname[:-4])

lists = [l_mig,l_anu,l_mid,l_ind,thumb,r_ind,r_mid,r_anu,r_mig]

with open('pushing.txt','w') as f:
    for i in range(len(lists)):
        for img in lists[i]: 
            f.write(img +' '+str(i) +'\n') 