

import cv2
import numpy as np
import matplotlib.pyplot as plt
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (image - min_val) * (255 / (max_val - min_val))
    
    
    return np.array(stretched_image, dtype=np.uint8)
def intensity_level_slicing(image, lower_threshold, upper_threshold, high_value=255, low_value=1):
    sliced_image = np.where((image >= lower_threshold) & (image <= upper_threshold), high_value, low_value)
    return np.array(sliced_image, dtype=np.uint8)


def bit_plane_slicing(image, bit_plane):
    # Shift the bits of the image to the right and get the last bit
    bit_sliced_image = (image >> bit_plane) & 1
    # Scale up to 0-255 range for visualization
    bit_sliced_image = bit_sliced_image * 255
    return np.array(bit_sliced_image, dtype=np.uint8)

image = cv2.imread("demo.png", cv2.IMREAD_GRAYSCALE)
contrast_stretched_image = contrast_stretching(image)
intensity_sliced_image = intensity_level_slicing(image, 100, 200)
bit_plane_image = bit_plane_slicing(image, 7)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(contrast_stretched_image, cmap='gray')
plt.title('Contrast Stretching')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(intensity_sliced_image, cmap='gray')
plt.title('Intensity Level Slicing')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(bit_plane_image, cmap='gray')
plt.title('Bit-Plane Slicing (Plane 4)')
plt.axis('off')
plt.show()