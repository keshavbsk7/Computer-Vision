import cv2
import matplotlib.pyplot as plt

image_path = "images.jfif"
image = cv2.imread(image_path)

rotate=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
flip=cv2.flip(image, 1)

plt.figure(figsize=(10, 5))


plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(rotate)

plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(flip)

plt.axis('off')

plt.tight_layout()
plt.show()

