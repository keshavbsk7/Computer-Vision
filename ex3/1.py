import cv2

img = cv2.imread("img.jpg")
print("Oringinal Image")
print(img)
alpha = 2 
beta = 30    

adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
print("ADjusted image")
print(adjusted)
cv2.imshow('Original Image', img)
cv2.imshow('Adjusted Image', adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
