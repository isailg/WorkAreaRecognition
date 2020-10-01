import numpy as np              
import cv2 as cv

#Global Variables
Wid= 1344   #widht of frame
Mid= Wid/2  #Middle of frame
Hei= 376    #Height of frame

#Capture of Stereo Camera
Cam=cv.VideoCapture(1)

if not Cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
      
      ret, frame=Cam.read()
      gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

      #Separation of frames
      imgL = gray[0:Hei, 0:Mid]
      imgR = gray[0:Hei, Mid:Wid]
      if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
      cv.imshow("Right",imgL)
      cv.imshow("Left",imgR)
      
      if cv.waitKey(1)&0xFF==ord('s'):
         cv.imwrite('fr1.png',imgR)
         cv.imwrite('fl1.png',imgL)

######################################
      if cv.waitKey(1)&0xFF==ord('q'):
         break
Cam.release()
cv.destroyAllWindows()
######################################



