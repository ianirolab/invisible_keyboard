import os
import cv2

imgs = []

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
    if (k == ord('1')):
        imgs.append(fname)
        print(fname)
    elif (k == ord('a')):
        i = i - 2
    
    i += 1

    cv2.destroyAllWindows()
    if (k == ord('2')):
        os.remove('./pictures/' + fname)
        os.remove('./results/' + fname[:-4])

imgs.sort()
print(imgs)
with open('pushing.txt','w') as f:
    for img in imgs: 
        f.write(img + '\n')