import cv2
import numpy as np

img = cv2.imread("crack.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret, img = cv2.threshold(gray, 70, 255, 0)



grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

sfs = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        sfs[i, j] = np.arctan2(grad_y[i, j], grad_x[i, j])
        


_, thresh = cv2.threshold(sfs, 0.5, 1.0, cv2.THRESH_BINARY)
thresh = thresh.astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)
thresh = cv2.erode(thresh, kernel, iterations=2)
thresh = cv2.dilate(thresh, kernel, iterations=2)

orig_img = img.copy()


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(orig_img, [contour], -1, (255, 255, 255), 2)


cv2.imshow('Crack Detection', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()