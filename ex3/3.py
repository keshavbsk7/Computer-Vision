import cv2
import numpy as np
from skimage import filters
img=cv2.imread("img.jpg",0)
_, simple_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Original Image', img)
cv2.imshow('Simple Thresholding', simple_thresh)
cv2.imshow('Adaptive Thresholding', adaptive_thresh)
cv2.imshow("Otsu's Thresholding", otsu_thresh)


cv2.waitKey(0)
cv2.destroyAllWindows()