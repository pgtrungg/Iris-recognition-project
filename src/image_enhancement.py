import cv2
from iris_nomalization import iris_normalization


def image_enhancement(img, image_path):
    img = iris_normalization(img, image_path)
    img = cv2.convertScaleAbs(img, 1.5, 2)
    cv2.imwrite('imageshow/EnhancedImage.bmp', img)
    return img