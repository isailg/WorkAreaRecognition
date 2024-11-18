import numpy as np              
import cv2 as cv
import time
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy 
from matplotlib import pyplot as plt

#Symbolic Pattern Recognition
class symbolic_pattern:
    def metaheuristic_optimization(self):
        return 'symbolic_pattern'

######################### Global Variables ############################
ts="%s"%time.strftime("%j %H:%M:%S")
seti=[26,27,28,29,30,31,32,33,34,35,36,37,38]
j=1


############################## Main ###################################
wb= xlwt.Workbook()
ws = wb.add_sheet('Feature Matching')

for list in seti:
############################ Read Images ##############################
   bx="%s"%seti[j-1]
   ff = cv.imread('../Recursos/'+bx+'.png',0)
   imgL = ff[0:720, 0:1280]
   imgR = ff[0:720, 1280:2560]


############### Brute-Force Matching with ORB Descriptors #############
   img= ff
   strt=time.clock()
   # Initiate ORB detector
   orb = cv.ORB_create()
   # find the keypoints and descriptors with ORB
   kp1, des1 = orb.detectAndCompute(imgL,None)
   kp2, des2 = orb.detectAndCompute(imgR,None)
   # create BFMatcher object
   bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
   # Match descriptors.
   matches = bf.match(des1,des2)
   # Sort them in the order of their distance.
   matches = sorted(matches, key = lambda x:x.distance)
   usrt=time.clock()-strt   
   # Draw first 10 matches.
   img3 = cv.drawMatches(imgL,kp1,imgR,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   
   cv.imwrite('/home/zero/Documentos/Algorithm/Resultados/ORB'+bx+'.png', img3)
   ws.write(j, 0, usrt)
   ws.write(j, 1, len(matches))
##################### BF with SIFT Desc ###############################
   strt=time.clock()
   # Initiate SIFT detector
   sift = cv.xfeatures2d.SIFT_create()
   # find the keypoints and descriptors with SIFT
   kp1, des1 = sift.detectAndCompute(imgL,None)
   kp2, des2 = sift.detectAndCompute(imgR,None)
   # BFMatcher with default params
   bf = cv.BFMatcher()
   matches = bf.knnMatch(des1,des2,k=2)
   usrt=time.clock()-strt
   # Apply ratio test
   good = []
   for m,n in matches:
       if m.distance < 0.75*n.distance:
           good.append([m])
   
   # cv.drawMatchesKnn expects list of lists as matches.
   img3 = cv.drawMatchesKnn(imgL,kp1,imgR,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   cv.imwrite('../Resultados/SIFT'+bx+'.png',img3)
   ws.write(j, 3, usrt)
   ws.write(j, 4, len(matches))

########################## FLANN based Matcher ####################### 
   img= ff
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
   matches = flann.knnMatch(des1,des2,k=2)
   usrt=time.clock()-strt
   # Need to draw only good matches, so create a mask
   matchesMask = [[0,0] for i in range(len(matches))]
   # ratio test as per Lowe's paper
   for i,(m,n) in enumerate(matches):
       if m.distance < 0.7*n.distance:
           matchesMask[i]=[1,0]
   draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matchesMask,
                     flags = cv.DrawMatchesFlags_DEFAULT)
   img3 = cv.drawMatchesKnn(imgL,kp1,imgR,kp2,matches,None,**draw_params)
   
   cv.imwrite('../Resultados/FLANN'+bx+'.png', img3)
   ws.write(j, 6, usrt)
   ws.write(j, 7, len(matches))
   
   j=j+1
wb.save('FM.xls')



