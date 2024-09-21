import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seeds):
    height, width = img.shape
    segmented = np.zeros((height, width), np.uint8)
    region = list(seeds)  # Start with both seed points
    seed_values = [img[seed] for seed in seeds]  # Get values at seed points

    while region:
        x, y = region.pop(0)
        if segmented[x, y] == 0:
            segmented[x, y] = 255  
           
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    if segmented[nx, ny] == 0:
                        # Check similarity with both seed values
                        if any(abs(int(img[nx, ny]) - int(seed_value)) < 15 for seed_value in seed_values):
                            region.append((nx, ny))

    return segmented

# Load the image
image = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)
seed_points = [(100, 100), (100, 200)]  # Example seed points (x, y)

# Perform Region Growing
segmented_image = region_growing(image, seed_points)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.show()
