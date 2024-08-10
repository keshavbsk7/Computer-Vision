# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 05:07:08 2024

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
mat=np.random.randint(2,size=(8,8))
print(mat)
plt.imshow(mat,cmap="gray")
plt.title("Binary Image")