from math import sqrt
from scipy.spatial import distance
import cv2
import numpy as np


def iris_segmentation(img):
    pupil = find_pupil(img)
    if pupil is None:
        print("no pupil")
        return None
    else:
        iris = find_iris(img, pupil)
        if iris is None:
            print("no iris")
            return None
        else:
            create_mask(img, pupil, iris)
            cv2.circle(img, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)
            cv2.circle(img, (iris[0], iris[1]), iris[2], (0, 255, 0), 2)
            #cv2.imshow("img", img)
    cv2.imwrite("imageshow//SegmentedImage.bmp",img)
    return img,(pupil[0],pupil[1]),pupil[2],(iris[0], iris[1]), iris[2]

def find_pupil(img):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.medianBlur(img, 9)
    # Increase the constrast
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=5,
                               maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            copy = img.copy()
            cv2.circle(copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.imshow("img", copy)

        # Find the smallest circle (assuming it's the pupil)
        min_circle = min(circles[0, :], key=lambda x: x[2])
        return min_circle
    else:
        return None


def find_iris(img, inner_circle):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.medianBlur(img, 9)
    # Increase the constrast
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=30, param2=1,
                               minRadius=round(inner_circle[2] * 1.6), maxRadius=round(inner_circle[2] * 2.7))

    if circles is not None:
        print("circles")
        circles = np.uint16(np.around(circles))
        # Find the circle that neighbours the inner circle

        neyghbours = [x for x in circles[0, :] if distance.euclidean((x[0],x[1]),(inner_circle[0],inner_circle[1])) < 4]
        if len(neyghbours) == 0:
            return None
        max_circle = max(neyghbours, key=lambda x: x[2])
        return max_circle

    else:
        print("no circles")
        return None


def create_mask(img, inner_circle, outer_circle):
    # mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # cv2.circle(mask, (outer_circle[0], outer_circle[1]), outer_circle[2], (255, 255, 255), -1)
    # cv2.circle(mask, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 0, 0), -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # increase constrast
    # img = cv2.convertScaleAbs(img, alpha=1.24, beta=0)
    # masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("masked_img", masked_img)
    #
    # x = np.logical_or(masked_img > 245, masked_img < 120)
    # masked_img[x] = 0
    # cv2.imshow("masked_img1", masked_img)
    sobel_horizontal = np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)))
    cv2.imshow("sobel_horizontal", sobel_horizontal)

    # median filter
    sobel_horizontal = cv2.medianBlur(sobel_horizontal, 5)
    cv2.imshow("sobel_horizontal", sobel_horizontal)

