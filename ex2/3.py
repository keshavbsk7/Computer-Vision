import cv2
img = cv2.imread(r"demo.png")

# Shrinking the image using nearest-neighbor interpolation
nearest = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

# Zooming the image using linear and cubic interpolation
linear = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
cubic = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Shrink Nearest', nearest)
cv2.imshow('Zoom Linear', linear)
cv2.imshow('Zoom Cubic', cubic)

cv2.waitKey(0)
cv2.destroyAllWindows()
