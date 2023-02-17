import cv2

key_map = {'sx_mig':['q','a','z'],'sx_anu':['w','s','x'],'sx_mid':['e','d','c'],'sx_ind':['r','t','f','g','c','v'],
          'dx_ind':['y','u','h','j','n','m'],'dx_mid':['i','k',','],'dx_anu':['o','l','.'],'dx_mig':['p',';','/']}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 2
lineType               = 2