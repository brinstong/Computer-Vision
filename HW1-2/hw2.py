import glob

import cv2;
import numpy as np;
import scipy as sp;
import os;
import sys;
import math;
from scipy.optimize import minimize




def canny_edge_detection(img):
    SIGMA = 3;
    THRESHOLD_LOW = 250;
    THRESHOLD_HIGH = 500;
    APPERTURE_SIZE = 5;

    img = cv2.GaussianBlur(img, (0,0), SIGMA)
    return cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH,apertureSize=APPERTURE_SIZE,L2gradient=True);

class line :

    point1 = [0,0]
    point2 = [0,0]

    def __init__(self,p1,p2):
        self.point1 = p1;
        self.point2 = p2;

    def getPoints(self):
        return self.point1, self.point2;


def find_intersection(line1, line2):
    x1 = line1.getPoints()[0][0]
    y1 = line1.getPoints()[0][1]
    x2 = line1.getPoints()[1][0]
    y2 = line1.getPoints()[1][1]


    x3 = line2.getPoints()[0][0]
    y3 = line2.getPoints()[0][1]
    x4 = line2.getPoints()[1][0]
    y4 = line2.getPoints()[1][1]

    xdiff = (x2 - x1, x4 - x3)
    ydiff = (y2 - y1, y4 - y3)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det([x2,y2],[x1,y1]), det([x4,y4],[x3,y3]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def probabilistic_hough_transform(edges, img) :

    RHO = 1
    THETA = 1 * np.pi/180
    THRESHOLD = 100
    LENGTH_MIN = 15
    LENGTH_MAX_GAP = 10

    houghp_lines = cv2.HoughLinesP(edges,RHO,THETA,THRESHOLD,minLineLength=LENGTH_MIN,maxLineGap=LENGTH_MAX_GAP)

    houghp_output = img.copy()

    h, w = img.shape[:2]

    diagonal_length = math.sqrt(h*h+w*w)

    ls = []

    for i in range(0,len(houghp_lines)-1):

        for x1, y1, x2, y2 in houghp_lines[i]:
            ln = line([x1, y1], [x2, y2])
            ls.append(ln)

            cv2.line(houghp_output, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return houghp_output, ls

def standard_hough_transform(edges, img) :

    RHO = 1
    THETA = 1 * np.pi/180
    THRESHOLD = 120

    hough_lines = cv2.HoughLines(edges,RHO,THETA,THRESHOLD)

    hough_output = img.copy()

    h, w = img.shape[:2]

    diagonal_length = math.sqrt(h*h+w*w)

    ls = []

    for i in range(0,len(hough_lines)-1):

        for rho, theta in hough_lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + diagonal_length * (-b))
            y1 = int(y0 + diagonal_length * (a))
            x2 = int(x0 - diagonal_length * (-b))
            y2 = int(y0 - diagonal_length * (a))

            l = line([x0, y0], [x1, y1])

            ls.append(l)

            cv2.line(hough_output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return hough_output, ls

def draw_extended_lines(img,ls):

    h, w = img.shape[:2]

    diagonal_length = math.sqrt(h*h+w*w)

    houghp_extended = img.copy()

    for ln in ls :
        x1 = ln.getPoints()[0][0]
        y1 = ln.getPoints()[0][1]

        x2 = ln.getPoints()[1][0]
        y2 = ln.getPoints()[1][1]

        theta = math.atan2(y2-y1,x2-x1)
        a = np.cos(theta)
        b = np.sin(theta)
        x1 = int(x1 + diagonal_length * a)
        y1 = int(y1 + diagonal_length * b)


        x2 = int(x2 - 2 * diagonal_length * a)
        y2 = int(y2 - 2 * diagonal_length * b)

        cv2.line(houghp_extended,(x1,y1),(x2,y2),(250,0,0),2)

    return houghp_extended


def draw_circles(img,ls):

    RADIUS = 7

    h,w = img.shape[:2]

    size = len(ls)
    circle_output = img.copy()
    circle_centers = []

    for i in range(0,size) :
        for j in range(i+1,size) :
            if find_intersection(ls[i],ls[j]) :
                x,y = find_intersection(ls[i],ls[j])
                if not (x<0 or x>w or y<0 or y>h):
                    cv2.circle(circle_output,(x,y),RADIUS,(0,0,255),2)
                    circle_centers.append([x,y])

    return circle_output, circle_centers

def is_point_on_line(point,line) :

    xp = int(point[0])
    yp = int(point[1])


    x1 = int(line.getPoints()[0][0])
    y1 = int(line.getPoints()[0][1])
    x2 = int(line.getPoints()[1][0])
    y2 = int(line.getPoints()[1][1])

    if (x2 - x1) == 0 :
        m =1
    else :
        m = (y2 - y1)/(x2 - x1)

    y = m * (xp - x1) + y1

    #if (y == yp and y >= min(y1,y2) and y<= max(y1,y2) and xp>= min(x1,x2) and xp<= max(x1,x2)) :
    if (y == yp):
        return True
    else:
        return False



def draw_vanishing_points(circle_centers, lines, image) :

    best_center = circle_centers[0]
    best_count = -1


    i = 0;

    for center in circle_centers :

        count = 0


        for line in lines :

            if (is_point_on_line(center,line)) :
                print (count)
                count = count + 1
                if count > best_count:
                    best_count = count
                    best_center = center

    cv2.circle(image, (best_center[0], best_center[1]), 5, (0, 0, 0), 10)
    cv2.circle(image, (best_center[0], best_center[1]), 20, (0,0,0), 5)

    return image





images =  glob.glob("images/*");
for image in images :

    filename = os.path.basename(image).split(".")[0];

    if not os.path.exists(filename):
        os.makedirs(filename)



    print(filename)

    img = cv2.imread(image)
    cv2.imwrite(filename+"/original_img.jpg",img)

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename+"/gray_img.jpg",gray)

    canny_edge = canny_edge_detection(gray);
    cv2.imwrite(filename+"/canny.jpg",canny_edge)

    hough_output,lines_hough = standard_hough_transform(canny_edge,img)
    cv2.imwrite(filename+"/hough.jpg",hough_output)

    houghp_output, lines_houghp = probabilistic_hough_transform(canny_edge,img)
    cv2.imwrite(filename+"/houghp.jpg",houghp_output)

    houghp_extended = draw_extended_lines(img,lines_houghp)
    cv2.imwrite(filename+"/houghp_extended.jpg",houghp_extended)

    circle_hough, circle_centers_hough = draw_circles(hough_output, lines_hough)
    cv2.imwrite(filename + "/hough_circle.jpg", circle_hough)

    circle_houghp, circle_centers_houghp = draw_circles(houghp_extended, lines_houghp)
    cv2.imwrite(filename + "/houghp_circle.jpg", circle_houghp)

    vanishing_point_hough = draw_vanishing_points(circle_centers_hough, lines_hough, circle_hough)
    cv2.imwrite(filename + "/vanishing_point_hough.jpg", vanishing_point_hough)

    vanishing_point_houghp = draw_vanishing_points(circle_centers_houghp, lines_houghp, circle_houghp)
    cv2.imwrite(filename+ "/vanishing_point_houghp.jpg",vanishing_point_houghp)













