import cv2
import numpy as np

img=cv2.imread("data/synthetic/image.jpeg")
img=cv2.resize(img,(512,512))

#s gives the side length of the square patch
s=90
#we give the top left corner of the patch in img
sy,sx=50,50 
dy,dx=300,350

forged=img.copy()

forged[dy:dy+s,dx:dx+s]=img[sy:sy+s,sx:sx+s]

mask=np.zeros(img.shape[:2],np.uint8)
mask[sy:sy+s,sx:sx+s]=255
mask[dy:dy+s,dx:dx+s]=255

cv2.imwrite("data/synthetic/forged.jpg",forged)
cv2.imwrite("data/synthetic/mask.jpg",mask)

print("shift vector is: ", dy-sy,dx-sx)


