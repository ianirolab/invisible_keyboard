import os
import cv2

imgs = []

names = os.listdir('./pictures')
names.sort()

i = 0
while (i < len(names)):
    try:
        fname = names[names.index(str(i) + '.png')]
        img = cv2.imread('./pictures/' + fname)
    except:
        continue
    if(img is None):
        continue

    cv2.imshow(fname, img)
    
    k = cv2.waitKey(0)
    if (k == ord('1')):
        imgs.append(fname)
        print(fname)
    elif (k == 81):
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