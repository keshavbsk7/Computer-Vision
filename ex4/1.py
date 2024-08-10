import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to plot histogram
def plot_histogram(image, title):
    plt.figure(figsize=(8, 6))
    plt.hist(image.ravel(), bins=256, range=(0, 256), density=True)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.grid(True)
    plt.show()

# Load images (replace with your image paths)
dark_image = cv2.imread('f.jpg', cv2.IMREAD_GRAYSCALE)

light_image = cv2.imread('g.png', cv2.IMREAD_GRAYSCALE)
low_contrast_image = cv2.imread('s.jpg', cv2.IMREAD_GRAYSCALE)
high_contrast_image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)

# Plot histograms for each type of image
plot_histogram(dark_image, 'dark_image')
plot_histogram(light_image, 'light_image')
plot_histogram(low_contrast_image, 'low_contrast_image ')
plot_histogram(high_contrast_image, 'high_contrast_image')
