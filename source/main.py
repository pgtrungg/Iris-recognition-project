import cv2
from Matching import matching
img1 = cv2.imread("dataset\\1\\right\\aevar4.bmp")
img2 = cv2.imread("dataset\\1\\right\\aevar3.bmp")
print(matching(img1, img2, 0.32))
