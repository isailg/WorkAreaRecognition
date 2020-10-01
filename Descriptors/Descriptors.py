import numpy as np              
import cv2 as cv
import time
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy 
from matplotlib import pyplot as plt

######################### Global Variables ############################
ts="%s"%time.strftime("%j %H:%M:%S")
seti=[26,27,28,29,30,31,32,33,34,35,36,37,38]
j=1


############################## Main ###################################
lb = open_workbook('Bit.xls')
wb=copy(lb)
ws = wb.get_sheet(0)

for list in seti:
############################ Read Images ##############################
   bx="%s"%seti[j-1]
   ff = cv.imread('../Recursos/'+bx+'.png',0)
   imgL = ff[0:720, 0:1280]
   imgR = ff[0:720, 1280:2560]



############################# SIFT Features ###########################
   img= ff
   strt=time.clock()
   sift = cv.xfeatures2d.SIFT_create()
   kp = sift.detect(img,None)
   usrt=time.clock()-strt
   iml=cv.drawKeypoints(img,kp,img)


   cv.imwrite('../Resultados/SIFT'+bx+'.png', iml)
   ws.write(j, 0, usrt)
   ws.write(j, 1, len(kp))
   
######################### FAST Feature Detector #######################
   img= ff
   strt=time.clock()
   # Initiate FAST object with default values
   fast = cv.FastFeatureDetector_create()
   # find and draw the keypoints
   kp = fast.detect(img,None)
   usrt=time.clock()-strt
   img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
   # Print all default params
   #print( "Threshold: {}".format(fast.getThreshold()) )
   #print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
   #print( "neighborhood: {}".format(fast.getType()) )
   #print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
   cv.imwrite('../Resultados/fast_true'+bx+'.png',img2)
   # Disable nonmaxSuppression
   fast.setNonmaxSuppression(0)
   kp = fast.detect(img,None)
   #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
   img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
   cv.imwrite('../Resultados/fast_false'+bx+'.png',img3)
   ws.write(j, 3, usrt)
   ws.write(j, 4, len(kp))

################################# BRIEF ###############################
   img= ff
   # Initiate FAST detector
   strt=time.clock()
   star = cv.xfeatures2d.StarDetector_create()
   # Initiate BRIEF extractor
   brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
   # find the keypoints with STAR
   kp = star.detect(img,None)
   # compute the descriptors with BRIEF
   kp, des = brief.compute(img, kp)
   usrt=time.clock()-strt
   #print( brief.descriptorSize() )
   #print( des.shape ) 
   cv.imwrite('../Resultados/BRIEF'+bx+'.png',img)
   ws.write(j, 6, usrt)
   ws.write(j, 7, len(kp))

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
   
   cv.imwrite('../Resultados/ORB'+bx+'.png', img3)
   kp=len(kp1)+len(kp2)
   ws.write(j, 9, usrt)
   ws.write(j, 10, kp)

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
   kp=len(kp1)+len(kp2)
   ws.write(j, 12, usrt)
   ws.write(j, 13, kp)

##################### Depth Map from Stereo Camera ####################
   strt=time.clock()
   stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
   disparity = stereo.compute(imgL,imgR)
   usrt=time.clock()-strt
   cv.imwrite('../Resultados/Stereo'+bx+'.png', disparity)
   ws.write(j, 15, usrt)
   #plt.imshow(disparity,'gray')
   #plt.show()

####################################################
   MIN_MATCH_COUNT = 10
   strt=time.clock()
   # Initiate SIFT detector
   sift = cv.xfeatures2d.SIFT_create()
   # find the keypoints and descriptors with SIFT
   kp1, des1 = sift.detectAndCompute(imgL,None)
   kp2, des2 = sift.detectAndCompute(imgR,None)
   FLANN_INDEX_KDTREE = 1
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks = 50)
   flann = cv.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(des1,des2,k=2)
   # store all the good matches as per Lowe's ratio test.
   good = []
   usrt=time.clock()-strt
   for m,n in matches:
       if m.distance < 0.7*n.distance:
           good.append(m)  

   if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
      matchesMask = mask.ravel().tolist()
      h,w = imgL.shape
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      dst = cv.perspectiveTransform(pts,M)
      imgR = cv.polylines(imgR,[np.int32(dst)],True,255,3, cv.LINE_AA)
   else:
      print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
      matchesMask = None


   params_draw = dict(matchColor = (0,255,0), # draw matches in green color
                      singlePointColor = None,
                      matchesMask = matchesMask, # draw only inliers
                      flags = 2)
   img3 = cv.drawMatches(imgL,kp1,imgR,kp2,good,None,flags=2)
   #plt.imshow(img3, 'gray'),plt.show()
   #print (good)

   cv.imwrite('../Resultados/Homography'+bx+'.png', img3)
   ws.write(j, 17, usrt)
   ws.write(j, 18, len(good))
   
   j=j+1
wb.save('Bit.xls')



