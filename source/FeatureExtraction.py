import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ImageEnhancement import image_enhancement

def apply_gabor_filters(image):
    image=image_enhancement(image)
    # Define Gabor filter parameters
    ksize = 21  # Filter size
    sigma = 5.0  # Standard deviation of the Gaussian
    theta = 0.0  # Orientation of the filter
    lambda_val = 10.0  # Wavelength of the sinusoidal function
    gamma = 0.5  # Spatial aspect ratio

    # Create Gabor filter
    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_val, gamma)

    # Convolve the iris image with the Gabor filter
    filtered_image = cv2.filter2D(image, cv2.CV_64F, gabor_filter)

    # Calculate the phase of the filtered image
    phase_image = np.angle(filtered_image)

    # Convert phase information to a binary code
    binary_iris_code = (phase_image > 0).astype(np.uint8)
    return binary_iris_code

#MAIN
input_path="dataset\\46\\left\\zulaikahl1.bmp"
img=cv2.imread(input_path)

# Apply Gabor filters and get features
gabor_features1 = apply_gabor_filters(img)

input_path="dataset\\46\\left\\zulaikahl2.bmp"
img=cv2.imread(input_path)

# Apply Gabor filters and get features
gabor_features2 = apply_gabor_filters(img)

cv2.imshow("1",gabor_features1*255)
cv2.imshow("2",gabor_features2*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

