import  cv2
from IrisNomalization import iris_normalization
def image_enhancement(img):
    img=iris_normalization(img)
    img = cv2.convertScaleAbs(img,1.5,2)
    cv2.imwrite('imageshow/EnhancedImage.bmp',img)
    return img
    
    
