import cv2
import matplotlib.pyplot as plt
# Load an image using OpenCV
image_path = "C:/Users/Administrator/Documents/17419/images.jfif"
image = cv2.imread(image_path)



cropped_image = image[100:200,50:100]


resized_image = cv2.resize(image, (100, 100))

plt.figure(figsize=(10, 5))


plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(cropped_image)
plt.title('Cropped Image')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(resized_image)
plt.title('Resized Image')
plt.axis('off')

plt.tight_layout()
plt.show()



