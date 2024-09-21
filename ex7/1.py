import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('images.jfif', cv2.IMREAD_GRAYSCALE)


_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


kernel = np.ones((5, 5), np.uint8)


eroded = cv2.erode(binary_image, kernel, iterations=1)


dilated = cv2.dilate(binary_image, kernel, iterations=1)


opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Erosion')
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Dilation')
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Opening')
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Closing')
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
