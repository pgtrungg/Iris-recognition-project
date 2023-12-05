import cv2
import numpy as np
import IrisSegmentation
import cv2
import numpy as np

def iris_normalization(image):
    segmented_iris,inner_center,inner_radius,outer_center,outer_radius=IrisSegmentation.iris_segmentation(image)
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)

    normalized_image = np.zeros_like(image)

    for (i, theta) in enumerate(thetas):
        for j in range(height):
            r = j / height

            Xin = inner_center[0] + inner_radius * np.cos(theta)
            Yin = inner_center[1] + inner_radius * np.sin(theta)
            Xo = outer_center[0] + outer_radius * np.cos(theta)
            Yo = outer_center[1] + outer_radius * np.sin(theta)

            Xc = int((1 - r) * Xin + r * Xo)
            Yc = int((1 - r) * Yin + r * Yo)

            normalized_image[j, i] = image[Yc, Xc]
    cv2.imwrite('imageshow/NormalizedImage.bmp',normalized_image)
    return normalized_image

