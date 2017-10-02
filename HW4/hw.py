import cv2
import numpy as np
import scipy as sp
import random
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac

def get_SIFT_features(img) :

    # input : Image file
    # output : keypoints, descriptors

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray,None)

    return keypoints, descriptors


def get_best_matches(descriptors1, descriptors2) :

    # input : descriptors of two image files
    # output : selected matches

    matches = cv2.BFMatcher().knnMatch(descriptors1, descriptors2, 2)
    selected_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return selected_matches

def draw_keypoints(keypoints, img) :
    # output : image with plotted keypoints

    img_out = img.copy()
    cv2.drawKeypoints(img,keypoints, img_out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_out


def draw_matches(img1, img2, matches, keypoints1, keypoints2) :
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view1 = img1.copy()
    view2 = img2.copy()
    view[:h1, :w1] = view1
    view[:h2, w1:] = view2

    for m in matches:
        color = tuple([sp.random.randint(0, 255) for _ in range(3)])
        cv2.line(view, (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1])), (int(keypoints2[m.trainIdx].pt[0] + w1), int(keypoints2[m.trainIdx].pt[1])),  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), lineType=cv2.LINE_8)

    return view


def get_Affine_Transformation(keypoints1, keypoints2, descriptors1, descriptors2, matches) :


    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    M, _ = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=2, max_trials=200)

    return M

def get_Homograph_transformation (keypoints1, keypoints2, descriptors1, descriptors2, matches) :
    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    M, status = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)

    return M, status


def get_points(keypoints) :

    point_list = [x.pt for x in keypoints]
    return point_list


def warp_images(img1, img2, homography):
    # Warps second image to plane of first image based on provided homography.

    # Warps img2 to img1.

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    points1 = np.array([[0, 0], [0, h1], [w1, 0], [w1, h1]],np.float32).reshape(-1,1,2)
    points2 = np.array([[0, 0], [0, h2], [w2, 0], [w2, h2]],np.float32).reshape(-1,1,2)
    points2_pers = cv2.perspectiveTransform(points2, homography)

    all_points = np.concatenate((points1,points2_pers))
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel())

#    print(x_min,y_min,x_max,y_max)

    smallest_coords = [-x_min, -y_min]
    homography_translation = np.array([[1, 0, smallest_coords[0]], [0, 1, smallest_coords[1]], [0, 0, 1]])

    x_size = x_max - x_min
    y_size = y_max - y_min

    homography_final = homography_translation.dot(homography)

    output_img = cv2.warpPerspective(img2, homography_final, (x_size, y_size))
    output_img[smallest_coords[1]:h1+smallest_coords[1], smallest_coords[0]:w1+smallest_coords[0]] = img1

    return output_img


def question_1() :
    '''
    img1 = cv2.imread("images/IMG_1188.JPG")
    img2 = cv2.imread("images/IMG_1189.JPG")
    '''

    '''
    img1 = cv2.imread("images/IMG_1200.JPG")
    img2 = cv2.imread("images/IMG_1201.JPG")
    '''


    img1 = cv2.imread("images/IMG_1210.JPG")
    img2 = cv2.imread("images/IMG_1211.JPG")



    kp1, des1 = get_SIFT_features(img1)
    kp2,des2 = get_SIFT_features(img2)
    matches = get_best_matches(des1,des2)
    img_out = draw_matches(img1,img2,matches,kp1,kp2)
    cv2.namedWindow('Question-1',cv2.WINDOW_NORMAL)
    cv2.imshow("Question-1",img_out)
    cv2.waitKey(0)




def question_2() :


    img1 = cv2.imread("images/IMG_1188.JPG")
    img2 = cv2.imread("images/IMG_1189.JPG")

    '''
    ------ Transformation Matrix - Affine ------
    [[  1.02333808e+00  -1.60811860e-02   1.85181549e+02]
     [  4.38329502e-04   1.00558400e+00   3.42286682e+00]]

    ------ Transformation Matrix - Homography ------
    [[  9.77620304e-01   1.54075064e-02  -1.81539749e+02]
     [ -3.50916205e-04   9.94283795e-01  -3.31968951e+00]
     [  8.19764239e-08  -1.58300352e-07   1.00000000e+00]]
    '''

    '''
    img1 = cv2.imread("images/IMG_1200.JPG")
    img2 = cv2.imread("images/IMG_1201.JPG")
    '''
    '''
    ------ Transformation Matrix - Affine ------
    [[  1.01881707e+00   2.10916959e-02  -3.25989594e+02]
     [ -1.08655766e-02   9.99276400e-01   6.10512733e+00]]

    ------ Transformation Matrix - Homography ------
    [[  7.59037316e-01  -2.20614355e-02   4.27962555e+02]
     [ -2.04153638e-02   9.02750790e-01   3.03729134e+01]
     [ -8.79734143e-05  -3.05624349e-06   1.00000000e+00]]
    '''

    '''
    img1 = cv2.imread("images/IMG_1210.JPG")
    img2 = cv2.imread("images/IMG_1211.JPG")
    '''
    '''
    ------ Transformation Matrix - Affine ------
    [[  1.07259655e+00  -1.49559788e-02  -7.87505737e+02]
     [  5.73794059e-02   1.00863004e+00  -1.36618805e+02]]

    ------ Transformation Matrix - Homography ------
    [[  4.61558968e-01  -8.99012107e-03   8.97355774e+02]
     [ -1.37915149e-01   8.11669171e-01   1.66353638e+02]
     [ -1.85847326e-04  -9.10070230e-06   1.00000000e+00]]
    '''

    kp1, des1 = get_SIFT_features(img1)
    kp2, des2 = get_SIFT_features(img2)

    matches = get_best_matches(des1, des2)

    tranformation_matrix_affine = get_Affine_Transformation(kp1,kp2,des1,des2, matches)
    print ("------ Transformation Matrix - Affine ------")
    print(np.float32(tranformation_matrix_affine._inv_matrix)[0:2,:])

    tranformation_matrix_homography, _ = get_Homograph_transformation(kp1,kp2,des1,des2, matches)
    print("------ Transformation Matrix - Homography ------")
    print(np.float32(tranformation_matrix_homography))



def question_3():
    '''
    img1 = cv2.imread("images/IMG_1188.JPG")
    img2 = cv2.imread("images/IMG_1189.JPG")
    '''

    '''
    img1 = cv2.imread("images/IMG_1200.JPG")
    img2 = cv2.imread("images/IMG_1201.JPG")
    '''

    '''
    img1 = cv2.imread("images/IMG_1210.JPG")
    img2 = cv2.imread("images/IMG_1211.JPG")
    '''

    img1 = cv2.imread("images/IMG_1211.JPG")
    img2 = cv2.imread("images/IMG_1212.JPG")

    kp1, des1 = get_SIFT_features(img1)
    kp2, des2 = get_SIFT_features(img2)

    matches = get_best_matches(des1, des2)

    tranformation_matrix_affine = get_Affine_Transformation(kp1, kp2, des1, des2, matches)
    tranformation_matrix_affine = np.float32(tranformation_matrix_affine._inv_matrix)[0:2, 0:3]
    print("------ Transformation Matrix - Affine ------")
    print(tranformation_matrix_affine)

    tranformation_matrix_homography, status = get_Homograph_transformation(kp1, kp2, des1, des2, matches)
    print("------ Transformation Matrix - Homography ------")
    print(np.float32(tranformation_matrix_homography))

    warped_homo_image = warp_images(img2, img1, tranformation_matrix_homography)
    cv2.namedWindow("Warped Homo Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped Homo Image", warped_homo_image)
    cv2.waitKey(0)


def question_4() :

    img1 = cv2.imread("images/IMG_1202.JPG")

    img2 = cv2.imread("images/IMG_1203.JPG")

    img3 = cv2.imread("images/IMG_1204.JPG")

    img4 = cv2.imread("images/IMG_1205.JPG")

    img5 = cv2.imread("images/IMG_1206.JPG")


    '''
    img1 = cv2.imread("images/IMG_1213.JPG")

    img2 = cv2.imread("images/IMG_1214.JPG")

    img3 = cv2.imread("images/IMG_1215.JPG")

    img4 = cv2.imread("images/IMG_1216.JPG")

    img5 = cv2.imread("images/IMG_1217.JPG")

    '''

    '''
    img1 = cv2.imread("images/IMG_1197.JPG")

    img2 = cv2.imread("images/IMG_1198.JPG")

    img3 = cv2.imread("images/IMG_1199.JPG")

    img4 = cv2.imread("images/IMG_1200.JPG")

    img5 = cv2.imread("images/IMG_1201.JPG")
    '''


    kp1, des1 = get_SIFT_features(img1)
    kp2, des2 = get_SIFT_features(img2)
    kp3, des3 = get_SIFT_features(img3)
    kp4, des4 = get_SIFT_features(img4)
    kp5, des5 = get_SIFT_features(img5)

    matches12 = get_best_matches(des1,des2)
    homo_mat_12, _ = get_Homograph_transformation(kp1,kp2,des1,des2,matches12)
    warped12 = warp_images(img2,img1,homo_mat_12)

    kp12, des12 = get_SIFT_features(warped12)
    matches12_3 = get_best_matches(des12, des3)
    homo_mat_12_3, _ = get_Homograph_transformation(kp12, kp3, des12, des3, matches12_3)
    warped12_3 = warp_images(img3,warped12, homo_mat_12_3)

    kp12_3, des12_3 = get_SIFT_features(warped12_3)
    matches123_4 = get_best_matches(des12_3, des4)
    homo_mat_123_4, _ = get_Homograph_transformation(kp12_3, kp4, des12_3, des4, matches123_4)
    warped123_4 = warp_images(img4, warped12_3,  homo_mat_123_4)


    kp1234, des1234 = get_SIFT_features(warped123_4)
    matches1234_5 = get_best_matches(des1234, des5)
    homo_mat_1234_5, _ = get_Homograph_transformation(kp1234, kp5, des1234, des5, matches1234_5)
    warped1234_5 = warp_images(img5, warped123_4, homo_mat_1234_5)

    cv2.namedWindow("5_Images", cv2.WINDOW_NORMAL)
    cv2.imshow("5_Images",warped1234_5)
    cv2.waitKey(0)

#question_1()
#question_2()
#question_3()
question_4()

