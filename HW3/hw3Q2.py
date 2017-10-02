import cv2
import sys
import scipy as sp
import random


file1 = "images/ST2MainHall4099.jpg"
file2 = "images/ST2MainHall4093.jpg"

file3 = "images/ST2MainHall4075.jpg"
file4 = "images/ST2MainHall4073.jpg"

file5 = "images/ST2MainHall4041.jpg"
file6 = "images/ST2MainHall4046.jpg"

file7 = "images/ST2MainHall4079.jpg"
file8 = "images/ST2MainHall4039.jpg"

file9 = "images/ST2MainHall4022.jpg"
file10 = "images/ST2MainHall4098.jpg"

'''
# Use this block instead if you want to take input images from arguments
if len(sys.argv) < 3:
    print 'usage: %s img1 img2' % sys.argv[0]
    sys.exit(1)

img1_path = sys.argv[1]
img2_path = sys.argv[2]

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
'''

img1 = cv2.imread(file9)
img2 = cv2.imread(file10)

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, d1 = sift.detectAndCompute(gray1,None)
kp2, d2 = sift.detectAndCompute(gray2,None)

matches = cv2.BFMatcher().knnMatch(d1,d2,2)

selected_matches = [m for m,n in matches if m.distance < 0.75*n.distance]

view1 = img1.copy()
view2 = img2.copy()

cv2.drawKeypoints(img1, kp1, view1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);
cv2.drawKeypoints(img2, kp2, view2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS);

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
view[:h1, :w1] = view1
view[:h2, w1:] = view2

for m in selected_matches:
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), (random.randint(0,255),random.randint(0,255),random.randint(0,255)),  lineType=cv2.LINE_8)

cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Matches", view)
cv2.waitKey()

cv2.imwrite("match5.jpg",view)
