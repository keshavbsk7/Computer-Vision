import cv2
import matplotlib.pyplot as plt


image_path = "images.jfif"  
image = cv2.imread(image_path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

r,g,b = cv2.split(image)


plt.figure(figsize=(10, 5))


plt.subplot(1, 3, 1)
plt.imshow(r, cmap='Reds')
plt.title('Red Channel')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(g, cmap='Greens')
plt.title('Green Channel')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(b, cmap='Blues')
plt.title('Blue Channel')
plt.axis("off")

plt.tight_layout()
plt.show()