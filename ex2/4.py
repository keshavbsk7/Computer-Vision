import cv2
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread("demo.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blurred_img = cv2.GaussianBlur(img,(5,5),1.5)

# Perform Canny edge detection
edges = cv2.Canny(blurred_img, 100, 200)

# Plotting the images using Matplotlib
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

# Show the plots
plt.show()
