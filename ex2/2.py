# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:05:12 2024

@author: keshavaram
"""

import cv2

img = cv2.imread(r"demo.png")

gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
median_blur = cv2.medianBlur(img, 5)
avg_blur = cv2.blur(img, (5, 5))

cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Average Blur', avg_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()