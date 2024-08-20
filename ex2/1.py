# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 07:40:46 2024

@author: keshavaram
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('demo.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images.jfif',cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_xor = cv2.bitwise_xor(img1, img2)

plt.figure(figsize=(10, 5))


plt.subplot(1, 3, 1)
plt.imshow(bitwise_and)

plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(bitwise_or)

plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(bitwise_xor)

plt.axis('off')

plt.tight_layout()
plt.show()
