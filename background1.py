import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outfile = cv2.VideoWriter("out.avi",fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
time.sleep(2)
bg = 0
image = "palace.webp"

for i in range(0,60) :
    res,bg = cap.read()
bg = np.flip(bg, axis=1)

while(cap.isOpened()) :
    ret,img = cap.read()
    if not ret :
        break
    img = np.flip(img, axis=1)
    frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l_black = np.array([0,0,0])
    u_black = np.array([105,105,105])
    mask1 = cv2.inRange(frame,l_black,u_black)
    mask2 = image
    mask1 = mask1 + mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img,img,mask = mask2)
    res2 = cv2.bitwise_and(bg,bg,mask = mask1)
    finaloutput = cv2.addWeighted(res1,1,res2,1,0)
    outfile.write(finaloutput)
    cv2.imshow("magic",finaloutput)
    cv2.waitKey(20000)

    
cap.release()
outfile.release()
cv2.destroyAllWindows()
