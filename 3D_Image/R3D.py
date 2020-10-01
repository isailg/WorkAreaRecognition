import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi

ts="%s"%time.strftime("%j %H:%M:%S")
seti=[10]
j=1
avx=0.35
avy=0.03
NYT=[]
#--------------------------------------
Baseline=12
DisFoc=640/np.tan(2*pi*55/360)
#--------------------------------------
for list in seti:
   bx="%s"%seti[j-1]
   frame = cv.imread('/home/zero/Documentos/Algorithm/Recursos/'+bx+'.png',0) 
   imgL = frame[0:720, 0:1280]
   imgR = frame[0:720, 1280:2560]
   original = cv.imread('/home/zero/Documentos/Algorithm/Recursos/'+bx+'.png',1)
   original = original[0:720, 0:1280]
   ff = cv.imread('/home/zero/Documentos/Algorithm/Recursos/'+bx+'.png',1)
   ff= ff[0:720, 0:1280]
#------------------------------------------------------------------
   strt=time.clock()
   # Initiate SIFT detector
   sift = cv.xfeatures2d.SIFT_create()
   # find the keypoints and descriptors with SIFT
   kp1, des1 = sift.detectAndCompute(imgL,None)
   kp2, des2 = sift.detectAndCompute(imgR,None)
   # FLANN parameters
   FLANN_INDEX_KDTREE = 1
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks=50)   # or pass empty dictionary
   flann = cv.FlannBasedMatcher(index_params,search_params)
   matches = flann.knnMatch(des1,des2,k=1)
   
#------------------------------------------------------------------
   siz=ff.shape
   dx=np.arange(1,siz[0]+1)
   dy=np.arange(1,siz[1]+1)
   marx=np.zeros([siz[0]])
   mary=np.zeros([siz[1]])
   ysup= siz[0]-1

   hsv= cv.cvtColor(ff,cv.COLOR_BGR2HSV)
   lower = np.array([25,25,25])#25 recuerda
   upper = np.array([80,255,255])
   mask = cv.inRange(hsv, lower, upper)
   res = cv.bitwise_and(ff,ff,mask= mask)
   frame= cv.cvtColor(res,cv.COLOR_BGR2GRAY)
   
   # Cambio en X
   for y in range(0,siz[0]-1):
      
      ant=frame[y,0]
      ach=0
      for x in range(0,siz[1]-1):
         dif=np.absolute(int(ant)-int(frame [y,x]))
         ant= frame [y,x]
         ach= ach + dif 
      marx[ysup]= ach  
      ysup=ysup-1
   
   # Cambio en Y
   for x in range(0,siz[1]-1):
      
      ant=frame[0,x]
      ach=0
      for y in range(0,siz[0]-1):
         dif=np.absolute(int(ant)-int(frame [y,x]))
         ant= frame [y,x]
         ach= ach + dif
      mary[x]= ach   
   j=j+1
   
   #Busqueda de pixeles
   mxx= avx*(max(marx))
   mxy= avy*(max(mary))

   ccpx=[]
   ccpy=[]
   for m in range (0,len(marx)-1):
      mcm=marx[m]
      if mcm>=mxx:
         ccpx.append(720-m)
   for n in range (0,len(mary)-1):
      mcm=mary[n]
      if mcm>=mxy:
         ccpy.append(n)
   #Mask
   mask=np.zeros([siz[0],siz[1]])
   for f in range (0,len(ccpx)):
      for g in range (0,len(ccpy)):
         mask[ccpx[f],ccpy[g]]=frame[ccpx[f],ccpy[g]]
   for h in range (0,720):
      for i in range (0,1280):
         if mask[h,i] == 0:
            ff[h,i]=([0,0,0])    
   
   
#------------------------------------------------------------------ 
   p2f1=cv.KeyPoint_convert(kp1)
   p2f2=cv.KeyPoint_convert(kp2)
   X=[]
   Y=[]
   Z=[]
   Xv=[]
   Yv=[]
   Zv=[]
   for ig in range (0,len(matches)):
      x1=int(p2f1[ig,0])
      y1=int(p2f1[ig,1]) 
      x2=int(p2f2[matches[ig][0].trainIdx,0])
      y2=int(p2f2[matches[ig][0].trainIdx,1])
      
      xp0=x1
      xp1=x2
      if np.absolute(y2-y1)<10 and (xp0-xp1)>0:
         dist= Baseline*DisFoc/(xp0-xp1)
         if dist>0 and dist<500:
            if y1 in ccpx:
               Xv.append(x1)
               Yv.append(y1)
               Zv.append(dist)

            else:
               X.append(x1)
               Y.append(y1)
               Z.append(dist)
   usrt=time.clock()-strt
   NYT.append(usrt)
#------------------------------------------------------------------
   for l in range (0,len(Y)-1):
      Y[l]=720-Y[l]
   for l in range (0,len(Yv)-1):
      Yv[l]=720-Yv[l]      
   
#------------------------------------------------------------------   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')  
   ax.scatter(X, Z, Y, c='r', marker='o')
   ax.scatter(Xv, Zv, Yv, c='g', marker='^')
   ax.set_xlabel('X-axis of Image')
   ax.set_ylabel('Z-axis Depth')
   ax.set_zlabel('Y-axis of Image')
    
   plt.show() ################  Revisar para plotear #################
   #plt.savefig('/home/zero/Documentos/Algorithm/3D Image/Plots/'+bx+'d R3D '+ts+'.png')     
   #cv.imwrite('/home/zero/Documentos/Algorithm/3D Image/Plots/'+bx+'a Original '+ts+'.png',original)
   #cv.imwrite('/home/zero/Documentos/Algorithm/3D Image/Plots/'+bx+'b HSV '+ts+'.png',res)
   #cv.imwrite('/home/zero/Documentos/Algorithm/3D Image/Plots/'+bx+'c VF '+ts+'.png',ff)
#--------------------------------------------------------------------
"""
NXT=np.arange(1,len(NYT)+1)
fig, axf =plt.subplots()
line1, = axf.plot(NXT, NYT)
axf.set_xlabel('Test Number')
axf.set_ylabel('Seconds')
axf.legend()
plt.xticks([1,2,3,4,5,6,7,8,9,10],
           ["1","2","3","4","5","6","7","8","9","10"])
plt.title('Processing Time')
plt.savefig('/home/zero/Documentos/Algorithm/3D Image/Processing Time'+ts+'.png')
"""
