import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

ts="%s"%time.strftime("%j %H:%M:%S")
seti=[1,2,3,4,5,6,7,8,9,10,11,12,13]
j=1
l=20
u=80
v=25
lu="%s"%l+"%s"%u+"%s"%v
for list in seti:
   bx="%s"%seti[j-1]
   frame = cv.imread('../Recursos/'+bx+'.png',1)
   frame= frame[0:720, 0:1280]
   hsv= cv.cvtColor(frame,cv.COLOR_BGR2HSV)
   lower = np.array([l,25,v])
   upper = np.array([u,255,255])

   mask = cv.inRange(hsv, lower, upper)
   
   res = cv.bitwise_and(frame,frame,mask= mask)
   
   ir = np.concatenate((frame,res),axis=0)
   cv.imwrite('/Green Color/HSV'+bx+' '+lu+' '+ts+'.png', ir)
   j=j+1

cv.destroyAllWindows() 



"""
cap= cv.VideoCapture(0)
while(1):

   _, frame=cap.read()
   
   hsv= cv.cvtColor(frame,cv.COLOR_BGR2HSV)
   
   lower = np.array([25,50,50])
   upper = np.array([73,255,255])
   
   mask = cv.inRange(hsv, lower, upper)
   res = cv.bitwise_and(frame,frame,mask= mask)
   
   cv.imshow('frame',frame)
   #cv.imshow('mask',mask)
   cv.imshow('res',res)
   k = cv.waitKey(5) & 0xFF
   if k == 27:
      break
"""

"""
frame = cv.imread('../Recursos/29.png',1)
hsv= cv.cvtColor(frame,cv.COLOR_BGR2HSV)
l=25
u=73
while(1):
   k = cv.waitKey(0) & 0xFF
    
   if k == 67:
      l=l+1
   if k == 86:
      l=l-1
   if k == 78:
      u=u+1
   if k == 77:
      u=u-1
  
   if u<60:
      u=180
   if u>180:
      u=60
   if l<20:
      l=110
   if l>110:
      l=20
   
   lower = np.array([l,50,50])
   upper = np.array([u,255,255])
   
   mask = cv.inRange(hsv, lower, upper)
   res = cv.bitwise_and(frame,frame,mask= mask)
   
   #ir = np.concatenate((frame,res),axis=0)
   cv.imshow('img',res)
   
   if k == 27:
      break
"""
  
