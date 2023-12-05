import cv2
import sys
import numpy as np
from math import sqrt
from scipy.spatial import distance
def iris_inner_bound_detection(img):
    original_image=img.copy()
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.bilateralFilter(img,5,10,100)
    # Increase the constrast
    img =cv2.convertScaleAbs(img, alpha=1.6, beta=30)
    
    #img= cv2.Canny(img, 10, 150)
    circles=cv2.HoughCircles(img,method=cv2.HOUGH_GRADIENT,dp=1,minDist=50, param1=100, param2=30, minRadius=10, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles)
        min_d=sys.maxsize
        for i in circles[0,:]:
            height,width=img.shape
            d=sqrt((width/2-i[0])**2+(height/2-i[1])**2)
            if d< min_d:
                min_d=d
                center = (i[0], i[1])
                radius = i[2]
        cv2.circle(original_image, center, radius, (255, 0, 0),2)  
        print((center,radius))           
    return original_image,center,radius
def iris_outer_bound_detection(img,inner_center,inner_radius):
    original_image=img.copy()
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.bilateralFilter(img,10,10,100)
    # Increase the constrast
    img =cv2.convertScaleAbs(img, alpha=1.6, beta=30)
    #img=cv2.equalizeHist(img)
    # Dectect edge by Canny
    img1= cv2.Canny(img, 100, 100)
    # highlight the edge
    img=cv2.add(img,img1,img1)

    circles=cv2.HoughCircles(img,method=cv2.HOUGH_GRADIENT,dp=0.5,minDist=50, param1=50, param2=30, minRadius=round(inner_radius*1.5), maxRadius=round(inner_radius*2.5))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles)
        # Find the circle that is nearest to the inner_centers
        min_d=sys.maxsize
        for i in circles[0,:]:
            d = distance.euclidean((i[0],i[1]),inner_center)
            if d< min_d:
                min_d=d
                center = (i[0], i[1])
                radius = i[2]
        cv2.circle(original_image, center, radius, (0, 0, 255),2)           
    return original_image,center,radius

def iris_segmentation(img):
    img1=img.copy()
    img,inner_center,inner_radius=iris_inner_bound_detection(img)
    img,outer_center,outer_radius=iris_outer_bound_detection(img,inner_center,inner_radius)

    # Make the mask
    mask = np.zeros_like(img)*255
    cv2.circle(mask, outer_center, outer_radius, (255, 255, 255), thickness=-1)
    cv2.circle(mask, inner_center, inner_radius, (0, 0, 0), thickness=-1)

    # Apply the mask
    segmented_iris = cv2.bitwise_and(img1, mask)

    cv2.imwrite('imageshow/SegmentedImage.bmp',segmented_iris)
    return segmented_iris,inner_center,inner_radius,outer_center,outer_radius




