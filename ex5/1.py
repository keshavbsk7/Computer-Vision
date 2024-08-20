# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:06:00 2024

@author: keshavaram
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
def image_negative(image):
    return 255 - image

def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image)
    return np.array(log_image, dtype=np.uint8)

def power_law_transformation(image, gamma):
    normalized_image = image / 255.0
    power_law_image = np.power(normalized_image, gamma)
    return np.uint8(power_law_image * 255)

image = cv2.imread("demo.png", cv2.IMREAD_GRAYSCALE)
negative_image = image_negative(image)
log_image = log_transformation(image)
gamma = 2.0
power_law_image = power_law_transformation(image, gamma)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(negative_image, cmap='gray')
plt.title('Negative Image')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(log_image, cmap='gray')
plt.title('Log Transformation')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(power_law_image, cmap='gray')
plt.title(f'Power-Law Transformation (Gamma = {gamma})')
plt.axis('off')
plt.show()