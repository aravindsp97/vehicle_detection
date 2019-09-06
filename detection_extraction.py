import numpy as np
import cv2
import  imutils

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

# Read the image file
image = cv2.imread('Car_image_10.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges

# find contours in the accumulated image, keeping only the largest
# ones

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]


# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

# image to string convesion
def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    image_file = Image.open(filename)
    image_file = image_file.convert('1')
    image_file = image_file.convert('L')
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

# cropping and extracting text region from bounded countour image
for i, ctr in enumerate(sorted_ctrs): 
    # Get bounding box 
    x, y, w, h = cv2.boundingRect(ctr) 
    
    # Getting ROI 
    roi = image[y:y+h, x:x+w] 
    # show ROI 
    #cv2.imshow('segment no:'+str(i),roi) 
    cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
    #cv2.waitKey(0) 
    if w > 15 and h > 15:
        a=[]
        cv2.imwrite('cropped.png'.format(i), roi)
        a.append('cropped.png'.format(i))
        
        for i in range(len(a)):
                for j in range(len(a)):
                        if len(ocr_core(a[i])) == len(ocr_core(a[j])):
                                print(ocr_core('cropped.png'.format(i)))
                                cv2.imshow('cropped.png'.format(j),roi)
                                break
cv2.waitKey(0) #Wait for user input before closing the images displayed

