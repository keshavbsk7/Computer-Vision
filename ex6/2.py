# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:57:06 2024

@author: keshavaram
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def bandpass_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    bandpass_filter = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if low_cutoff <= distance <= high_cutoff:
                bandpass_filter[i, j] = 1

    filtered_transform = f_transform_centered * bandpass_filter
    filtered_transform = np.fft.ifftshift(filtered_transform)
    restored_image = np.fft.ifft2(filtered_transform)
    return np.abs(restored_image)

def bandreject_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    bandreject_filter = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if low_cutoff <= distance <= high_cutoff:
                bandreject_filter[i, j] = 0

    filtered_transform = f_transform_centered * bandreject_filter
    filtered_transform = np.fft.ifftshift(filtered_transform)
    restored_image = np.fft.ifft2(filtered_transform)
    return np.abs(restored_image)

def notch_filter(image, notch_centers, radius):
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    notch_filter = np.ones((rows, cols), np.float32)
    for notch in notch_centers:
        y, x = int(notch[0]), int(notch[1])
        Y, X = np.ogrid[:rows, :cols]
        mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= radius
        notch_filter[mask] = 0

    filtered_transform = f_transform_centered * notch_filter
    filtered_transform = np.fft.ifftshift(filtered_transform)
    restored_image = np.fft.ifft2(filtered_transform)
    return np.abs(restored_image)

def optimum_notch_filter(image, notch_centers, radius):
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    # Create a notch filter
    notch_filter = np.ones((rows, cols), np.float32)

    # Add each notch by zeroing out frequencies in the vicinity of the notch center
    for notch in notch_centers:
        y, x = int(notch[0]), int(notch[1])
        Y, X = np.ogrid[:rows, :cols]
        # Create a mask for the notch filter
        mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= radius
        notch_filter[mask] = 0

    # Apply the notch filter to the frequency domain image
    filtered_transform = f_transform_centered * notch_filter
    filtered_transform = np.fft.ifftshift(filtered_transform)
    restored_image = np.fft.ifft2(filtered_transform)
    return np.abs(restored_image)


def create_degradation_kernel(size, length):
    kernel = np.zeros(size)
    kernel[size[0] // 2, size[1] // 2] = 1
    kernel = cv2.filter2D(kernel, -1, np.ones((1, length)) / length)
    return kernel

def inverse_filter(image, H):
    rows, cols = image.shape
    f_transform = np.fft.fft2(image)
    f_transform_centered = np.fft.fftshift(f_transform)

    # Ensure that the kernel H has the same size as the image
    H_padded = np.zeros((rows, cols), np.float32)
    H_padded[:H.shape[0], :H.shape[1]] = H
    H_transform = np.fft.fft2(H_padded)
    H_transform_centered = np.fft.fftshift(H_transform)

    # Avoid division by zero by adding a small constant
    epsilon = 1e-6
    H_transform_centered = np.where(np.abs(H_transform_centered) < epsilon, epsilon, H_transform_centered)

    # Perform the inverse filtering
    G_transform = f_transform_centered / H_transform_centered
    G_transform = np.fft.ifftshift(G_transform)
    restored_image = np.fft.ifft2(G_transform)
    return np.abs(restored_image)
def wiener_filter(image, kernel, noise_var, estimated_noise_var):
    rows, cols = image.shape

    # Fourier transform of the kernel
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    kernel_ft_conj = np.conj(kernel_ft)

    # Fourier transform of the image
    image_ft = np.fft.fft2(image)

    # Compute the power spectrum of the kernel
    H_H_conj = np.abs(kernel_ft)**2

    # Compute the Wiener filter
    # Avoid division by zero by adding a small constant
    epsilon = 1e-6
    wiener_filter = (H_H_conj / (H_H_conj + noise_var / estimated_noise_var + epsilon))

    # Apply the Wiener filter to the frequency domain image
    restored_image_ft = image_ft * wiener_filter
    restored_image_ft = np.fft.ifftshift(restored_image_ft)

    # Inverse FFT to get the spatial domain image
    restored_image = np.fft.ifft2(restored_image_ft)

    return np.abs(restored_image)

# Load image
image = cv2.imread(r"img.jpg", cv2.IMREAD_GRAYSCALE)

# Apply various filters
restored_image_bandpass = bandpass_filter(image, 30, 60)
restored_image_bandreject = bandreject_filter(image, 30, 60)
notch_centers = [(100, 100), (150, 150)]
restored_image_notch = notch_filter(image, notch_centers, 10)
restored_image_optimum_notch = optimum_notch_filter(image, notch_centers, 10)
H = create_degradation_kernel((256, 256), 30)
restored_image_inverse = inverse_filter(image, H)
noise_var = 0.1
estimated_noise_var = 0.1
restored_image_wiener = wiener_filter(image, H, noise_var, estimated_noise_var)


# Display results
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title('Bandpass Filtered Image')
plt.imshow(restored_image_bandpass, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title('Bandreject Filtered Image')
plt.imshow(restored_image_bandreject, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title('Notch Filtered Image')
plt.imshow(restored_image_notch, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.title('Optimum Notch Filtered Image')
plt.imshow(restored_image_optimum_notch, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.title('Inverse Filtered Image')
plt.imshow(restored_image_inverse, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.title('Wiener Filtered Image')
plt.imshow(restored_image_wiener, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()