import numpy as np
import cv2  # You may need to install OpenCV: pip install opencv-python

def gaborconvolve(im, minWaveLength, mult, sigmaOnf):
    """
    Convolve each row of an image with 1D log-Gabor filters.

    Args:
        im: The image to be convolved.
        minWaveLength: Wavelength of the basis filter.
        mult: Multiplicative factor between each filter.
        sigmaOnf: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's transfer
                 function in the frequency domain to the filter center frequency.

    Returns:
        filterbank: The 1D array of complex-valued convolution results.
    """
    rows, ndata = im.shape
    logGabor = np.zeros(ndata)
    filterbank = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    wavelength = minWaveLength
    fo = 1 / wavelength
    logGabor[0 : int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
    logGabor[0] = 0

    for r in range(rows):
        signal = im[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterbank[r, :] = np.fft.ifft(imagefft * logGabor)

    return filterbank


def encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf):
    """
    Generate iris template and noise mask from the normalized iris region.

    Args:
        polar_array: Normalized iris region.
        noise_array: Normalized noise region.
        minWaveLength: Base wavelength.
        mult: Multiplicative factor between each filter.
        sigmaOnf: Bandwidth parameter.

    Returns:
        template: The binary iris biometric template.
        mask: The binary iris noise mask.
    """
    filterbank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

    length = polar_array.shape[1]
    template = np.zeros([polar_array.shape[0], 2 * length])
    h = np.arange(polar_array.shape[0])

    mask = np.zeros(template.shape)
    eleFilt = filterbank[:, :]

    H1 = np.real(eleFilt) > 0
    H2 = np.imag(eleFilt) > 0
    H3 = np.abs(eleFilt) < 0.0001

    for i in range(length):
        ja = 2 * i
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]
        mask[:, ja] = noise_array[:, i] | H3[:, i]
        mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

    return template, mask


def extract_iris_features(normalized_iris_image):
    """
    Extract iris features from a normalized iris image.

    Args:
        normalized_iris_image: Normalized iris image in grayscale.

    Returns:
        iris_features: The binary iris biometric template.
        noise_mask: The binary iris noise mask.
    """
    # Set Gabor filter parameters
    min_wave_length = 18
    mult = 1
    sigma_onf = 0.5

    # Create a placeholder for the noise array (all zeros)
    noise_array = np.zeros_like(normalized_iris_image)

    # Encode features using Gabor filters
    iris_features, noise_mask = encode(normalized_iris_image, noise_array, min_wave_length, mult, sigma_onf)

    return iris_features, noise_mask

# Example usage:
# Load normalized iris image (replace this path with your actual file path)
normalized_iris_image = cv2.imread('imageshow\\EnhancedImage.bmp', cv2.IMREAD_GRAYSCALE)

# Perform feature extraction
iris_features, noise_mask = extract_iris_features(normalized_iris_image)
print(iris_features)
# Display the results or save them as needed
cv2.imshow('Iris Template', iris_features * 255)  # Assuming binary, scaling for display
cv2.imshow('Noise Mask', noise_mask * 255)  # Assuming binary, scaling for display
cv2.waitKey(0)
cv2.destroyAllWindows()
