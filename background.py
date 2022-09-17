import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outfile = cv2.VideoWriter("out.avi",fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
time.sleep(2)
image = 0

for i in range(0,60) :
    ret,image = cap.read()
image = np.flip(image, axis=1)

while(cap.isOpened()) :
    ret,img = cap.read()
    if not ret :
        break
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_imageR2HSV)
    l_black = np.array([0,0,0])
    u_black = np.array([105,105,105])
    
    mask = cv2.inRange(hsv,l_black,u_black)
    l_black = np.array([30,30,0])
    u_black = np.array([104,153,70])
    
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask = cv2.bitwise_not(mask)
    res1 = cv2.bitwise_and(img,img,mask = mask)
    res2 = cv2.bitwise_and(image,image,mask = mask)
    finaloutput = cv2.addWeighted(res1,1,res2,1,0)
    outfile.write(finaloutput)
    cv2.imshow("magic",finaloutput)
    cv2.waitKey(20000)

cap.release()
outfile.release()
cv2.destroyAllWindows()

