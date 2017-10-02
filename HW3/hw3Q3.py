import cv2
import scipy as sp
import numpy as np
import sys
import glob
import csv
import os

def match(img1_path, img2_path):
    img1_c = cv2.imread(img1_path)
    img2_c = cv2.imread(img2_path)

    img1 = cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    matches = cv2.BFMatcher().knnMatch(d1, d2, 2)

    sel_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return float(len(sel_matches)) / min(len(k1), len(k2))


#output = match("images/ST2MainHall4099.jpg","images/ST2MainHall4099.jpg")
#print output

def main(argv):

    '''
    if len(argv) < 2:
        print 'usage: %s images_folder' % sys.argv[0]
        sys.exit(1)
    '''
    imgs = glob.glob("/home/brinstongonsalves/Documents/PyCharm/CV/images/" + "*.jpg")

    count = len(imgs)

    mat = np.zeros((count, count), np.float32);

    for i in range(0, count):
        for j in range(i + 1, count):
            m = match(imgs[i], imgs[j])
            print "%s %s %.3f \n" % (os.path.basename(imgs[i]), os.path.basename(imgs[j]),m),
            mat[i, j] = mat[j, i] = m

    mat = mat / mat.max()
    for i in range(0, count): mat[i, i] = 1.0
    np.savetxt('mat.txt',mat)

    table = np.loadtxt('/home/brinstongonsalves/Documents/PyCharm/CV/mat.txt')
    cv2.imwrite("allMatches1.png", table * 255)

if __name__ == "__main__":
    sys.exit(main(sys.argv));

# main("/home/brinstongonsalves/Documents/PyCharm/CV/images")