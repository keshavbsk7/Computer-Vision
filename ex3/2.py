# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:15:36 2024

@author: Administrator
"""

import cv2
import numpy as np
img=cv2.imread("images.jfif")
img=cv2.GaussianBlur(img,(5,5),1.5)
kernal=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharp=cv2.filter2D(img,-2, kernal)
cv2.imshow("Normal image", img)
cv2.imshow("sharpen image", sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()