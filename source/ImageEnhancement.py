import  cv2
from IrisNomalization import iris_normalization
def image_enhancement(img):
    img=iris_normalization(img)
    equalized_image = cv2.equalizeHist(img)
    cv2.imwrite('imageshow/EnhancedImage.bmp',equalized_image)
    return equalized_image
    
    
#MAIN
input_path="dataset\\1\\left\\aeval1.bmp"
img=cv2.imread(input_path)
img1=image_enhancement(img)
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows() 
