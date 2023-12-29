import  cv2
from IrisNomalization import iris_normalization
def image_enhancement(img):
    equalized_image = cv2.equalizeHist(img)
    cv2.imwrite('imageshow/EnhancedImage.bmp',equalized_image)
    return equalized_image
    
    
