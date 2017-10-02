import cv2

camera = cv2.VideoCapture(0)

location="/home/brinstongonsalves/Documents/PyCharm/CS682/HW1/"

ret, original_image = camera.read()
cv2.imwrite(location+"orig_img.jpeg",original_image)

gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(location+"gray_img.jpeg",gray_image)

median_orig_image_5 = cv2.medianBlur(original_image, 5)
cv2.imwrite(location+"median_orig_5.jpeg",median_orig_image_5)

median_orig_image_3 = cv2.medianBlur(original_image, 3)
cv2.imwrite(location+"median_orig_3.jpeg",median_orig_image_3)

median_gray_image_5 = cv2.medianBlur(gray_image, 5)
cv2.imwrite(location+"median_gray_5.jpeg",median_gray_image_5)

median_gray_image_3 = cv2.medianBlur(gray_image, 3)
cv2.imwrite(location+"median_gray_3.jpeg",median_gray_image_3)

gaussian_orig_1 = cv2.GaussianBlur(original_image,(7,7),1)
cv2.imwrite(location+"gaussian_orig_1.jpeg",gaussian_orig_1)

gaussian_orig_2 = cv2.GaussianBlur(original_image,(13,13),2)
cv2.imwrite(location+"gaussian_orig_2.jpeg",gaussian_orig_2)

gaussian_orig_3 = cv2.GaussianBlur(original_image,(19,19),3)
cv2.imwrite(location+"gaussian_orig_3.jpeg",gaussian_orig_3)

gaussian_gray_1 = cv2.GaussianBlur(gray_image,(7,7),1)
cv2.imwrite(location+"gaussian_gray_1.jpeg",gaussian_gray_1)

gaussian_gray_2 = cv2.GaussianBlur(gray_image,(13,13),2)
cv2.imwrite(location+"gaussian_gray_2.jpeg",gaussian_gray_2)

gaussian_gray_3 = cv2.GaussianBlur(gray_image,(19,19),3)
cv2.imwrite(location+"gaussian_gray_3.jpeg",gaussian_gray_3)

sobel_orig_x = cv2.Sobel(original_image,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_orig_x.jpeg",sobel_orig_x)

sobel_orig_y = cv2.Sobel(original_image,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_orig_y.jpeg",sobel_orig_y)

sobel_orig = cv2.addWeighted(sobel_orig_x,0.5,sobel_orig_y,0.5,1)
cv2.imwrite(location+"sobel_orig.jpeg",sobel_orig)

sobel_gray_x = cv2.Sobel(gray_image,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_gray_x.jpeg",sobel_gray_x)

sobel_gray_y = cv2.Sobel(gray_image,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_gray_y.jpeg",sobel_gray_y)

sobel_gray = cv2.addWeighted(sobel_gray_x,0.5,sobel_gray_y,0.5,1)
cv2.imwrite(location+"sobel_gray.jpeg",sobel_gray)

sobel_orig_gaussian_1_x = cv2.Sobel(gaussian_orig_1,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_orig_gaussian_1_x.jpeg",sobel_orig_gaussian_1_x)

sobel_orig_gaussian_1_y = cv2.Sobel(gaussian_orig_1,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_orig_gaussian_1_y.jpeg",sobel_orig_gaussian_1_y)

sobel_orig_gaussian_1 = cv2.addWeighted(sobel_orig_gaussian_1_x,0.5,sobel_orig_gaussian_1_y,0.5,1)
cv2.imwrite(location+"sobel_orig_gaussian_1.jpeg",sobel_orig_gaussian_1)

sobel_orig_gaussian_2_x = cv2.Sobel(gaussian_orig_2,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_orig_gaussian_2_x.jpeg",sobel_orig_gaussian_2_x)

sobel_orig_gaussian_2_y = cv2.Sobel(gaussian_orig_2,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_orig_gaussian_2_y.jpeg",sobel_orig_gaussian_2_y)

sobel_orig_gaussian_2 = cv2.addWeighted(sobel_orig_gaussian_2_x,0.5,sobel_orig_gaussian_2_y,0.5,1)
cv2.imwrite(location+"sobel_orig_gaussian_2.jpeg",sobel_orig_gaussian_2)

sobel_orig_gaussian_3_x = cv2.Sobel(gaussian_orig_3,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_orig_gaussian_3_x.jpeg",sobel_orig_gaussian_3_x)

sobel_orig_gaussian_3_y = cv2.Sobel(gaussian_orig_3,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_orig_gaussian_3_y.jpeg",sobel_orig_gaussian_3_y)

sobel_orig_gaussian_3 = cv2.addWeighted(sobel_orig_gaussian_3_x,0.5,sobel_orig_gaussian_3_y,0.5,1)
cv2.imwrite(location+"sobel_orig_gaussian_3.jpeg",sobel_orig_gaussian_3)

sobel_gray_gaussian_1_x = cv2.Sobel(gaussian_gray_1,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_gray_gaussian_1_x.jpeg",sobel_gray_gaussian_1_x)

sobel_gray_gaussian_1_y = cv2.Sobel(gaussian_gray_1,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_gray_gaussian_1_y.jpeg",sobel_gray_gaussian_1_y)

sobel_gray_gaussian_1 = cv2.addWeighted(sobel_gray_gaussian_1_x,0.5,sobel_gray_gaussian_1_y,0.5,1)
cv2.imwrite(location+"sobel_gray_gaussian_1.jpeg",sobel_gray_gaussian_1)

sobel_gray_gaussian_2_x = cv2.Sobel(gaussian_gray_2,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_gray_gaussian_2_x.jpeg",sobel_gray_gaussian_2_x)

sobel_gray_gaussian_2_y = cv2.Sobel(gaussian_gray_2,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_gray_gaussian_2_y.jpeg",sobel_gray_gaussian_2_y)

sobel_gray_gaussian_2 = cv2.addWeighted(sobel_gray_gaussian_2_x,0.5,sobel_gray_gaussian_2_y,0.5,1)
cv2.imwrite(location+"sobel_gray_gaussian_2.jpeg",sobel_gray_gaussian_2)

sobel_gray_gaussian_3_x = cv2.Sobel(gaussian_gray_3,cv2.CV_64F,1,0)
cv2.imwrite(location+"sobel_gray_gaussian_3_x.jpeg",sobel_gray_gaussian_3_x)

sobel_gray_gaussian_3_y = cv2.Sobel(gaussian_gray_3,cv2.CV_64F,0,1)
cv2.imwrite(location+"sobel_gray_gaussian_3_y.jpeg",sobel_gray_gaussian_3_y)

sobel_gray_gaussian_3 = cv2.addWeighted(sobel_gray_gaussian_3_x,0.5,sobel_gray_gaussian_3_y,0.5,1)
cv2.imwrite(location+"sobel_gray_gaussian_3.jpeg",sobel_gray_gaussian_3)

camera.release()
