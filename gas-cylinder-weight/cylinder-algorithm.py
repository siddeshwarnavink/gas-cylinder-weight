import cv2
from matplotlib import pyplot as plt

img= cv2.imread('./images/Cylinder_image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

_, binary_image = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image)
plt.show()