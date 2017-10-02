import cv2
import numpy as np
import scipy as sp


img = cv2.imread("images/ST2MainHall4019.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

print "------------SIFT Keypoints--------------"
print("Number of Keypoints : {0}".format(len(kp)))

for i in range(0,len(kp)) :
    print("Keypoint Coordinates : {0}".format(kp[i].pt))
    print("Keyoint Size : {0}".format(kp[i].size))
    print("Keypoint Angle : {0}".format(kp[i].angle))

sift_output = img.copy()
cv2.drawKeypoints(gray,kp,sift_output,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("sift.jpg", sift_output)


surf_output = img.copy()
surf = cv2.xfeatures2d.SURF_create(400)
kp = None
kp = surf.detect(gray,None)

print "------------SURF Keypoints--------------"
print("Number of Keypoints : {0}".format(len(kp)))

for i in range(0,len(kp)) :
    print("Keypoint Coordinates : {0}".format(kp[i].pt))
    print("Keyoint Size : {0}".format(kp[i].size))
    print("Keypoint Angle : {0}".format(kp[i].angle))

cv2.drawKeypoints(gray,kp,surf_output,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("surf.jpg",surf_output)
