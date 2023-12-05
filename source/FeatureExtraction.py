import cv2
import numpy as np
from scipy.ndimage import convolve
from ImageEnhancement import image_enhancement

def apply_gabor_filters(image):
    image=image_enhancement(image)
    # Set Gabor filter parameters
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.1, 0.5, 1, 2]
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to be in the range [0, 1]
    image = image / 255.0

    features = []

    for frequency in frequencies:
        for theta in orientations:
            # Create Gabor filter kernel
            kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=theta, lambd=10, gamma=0.5, psi=0)

            # Convolve the image with the Gabor kernel
            gabor_filtered = convolve(image, kernel, mode='constant', cval=0.0)

            # Extract feature statistics (mean and variance) from the filtered image
            mean = np.mean(gabor_filtered)
            variance = np.var(gabor_filtered)

            features.extend([mean, variance])

    return features

#MAIN
input_path="dataset\\1\\left\\aeval1.bmp"
img=cv2.imread(input_path)

# Apply Gabor filters and get features
gabor_features = apply_gabor_filters(img)

# Print or use the extracted features as needed
print("Gabor Features:", gabor_features)
