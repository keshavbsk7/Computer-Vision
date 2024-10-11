import cv2
import numpy as np
import matplotlib.pyplot as plt


left_img = cv2.imread('l.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('r.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if left_img is None or right_img is None:
    print("Error: Cannot load images.")
    exit()


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


disparity = stereo.compute(left_img, right_img)

disparity = cv2.normalize(disparity, disparity, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
disparity=cv2.equalizeHist(disparity)#improve visibilty

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(left_img, cmap='gray')
plt.title('Left Image')

plt.subplot(1, 3, 2)
plt.imshow(right_img, cmap='gray')
plt.title('Right Image')

plt.subplot(1, 3, 3)
plt.imshow(disparity, cmap='gray')
plt.title('Disparity Map')

plt.show()
