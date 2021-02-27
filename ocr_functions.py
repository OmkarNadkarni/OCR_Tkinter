import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import re

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def blur(img,KERNEL_SIZE,GAUSSIAN=True):
    if GAUSSIAN:
        return cv2.GaussianBlur(img, (KERNEL_SIZE,KERNEL_SIZE),0)
    else:
        return cv2.medianBlur(img,KERNEL_SIZE)


#adaptive thresholding for when lighting conditions are different
def thresholding(img, isAdaptive=0,TH=70):
    if isAdaptive==0:
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,81)
    elif isAdaptive==1:
        ret,img = cv2.threshold(img,TH,255,cv2.THRESH_BINARY)
    else:
        ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU )  #otsu binarization
    return img

#removes noise after thresholding
def noise_rem(img):
    return cv2.fastNlMeansDenoising(img,10,7,21)

#Used for thinning of text for better recognition
def erosion(img,kernel,no_iter):
    kernel = np.ones((kernel,kernel),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = no_iter)
    return erosion

#used to thicken the text in certain cases
def dilation(img,kernel,no_iter):
    kernel = np.ones((kernel,kernel),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = no_iter)
    return dilation

#returns text
def returnText(img):
    text = pytesseract.image_to_string(img, config=custom_config)
    text = re.sub('[^A-Z0-9]+', '', text)
    return text

#returns image with bounding box around the text in the given image
def imgBbox(img):
    boxes = pytesseract.image_to_boxes(img)
    h  = img.shape[0]
    w= img.shape[1]
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return img

#the goal is to resize the image to 300 dpi as it works really well for most cases
def ImageResize(path):
    temp_path = r'D:\barclays\ocr-test\test1_dpi.jpg'
    im = Image.open(path)
    width, height = im.size
    factor = min(1, float(1024.0 / height))
    size = int(factor * width), int(factor * height)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save(temp_path, dpi=(300, 300))
    new_img = cv2.imread(temp_path)
    return new_img

'''
1. RESIZE - convert image to 300 dpi as it works really well for most cases
2. GRAYSCALE - convert BGR image to grayscale
3. BLURRING - helps to smoothen the image
4. THRESHOLDING - converts img to binary
5. NOISE REMOVAL - after binary conversion there could be noise in many cases. Helps remove those
6. EROSION - helps with thinning of text
7. DILATION - helps to thicken the text
8. RETURN TEXT - returns text extracted by pytesseract
9. RETURN BOUNDING BOXES AROUND TEXT -
'''


pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 6'
# img_path = r'D:\barclays\ocr-test\captured-image-1.jpg'
# img_l = []
# img = cv2.imread(img_path)
# #img = Image.open(img_path)
# img_l.append(img)
# #img = resize(img,1900)
# img = ImageResize(img_path)
# img = grayscale(img)
# img_l.append(img)
# img = blur(img,3)
# img_l.append(img)
# img = thresholding(img,False)
# img_l.append(img)
# img = noise_rem(img)
# img_l.append(img)
# #img = erosion(img,3,1)
# #img = dilation(img,3,1)
# img_l.append(img)
#
# text = returnText(img)
# ht,wd = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split(" ")
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img,(x,ht-y),(w,ht-h),(180),2)
#     cv2.putText(img,b[0],(x,ht-y),cv2.FONT_HERSHEY_COMPLEX,1,(180),2)
# print(text)
# plt.imshow(img,cmap='gray')
# plt.show()
# # img2 = img#resize(img,400)
# # l1 = ['original image','grayscale','gaussianBlur','thresholding','noise removal','erosion']
# # width=10
# # height =10
# # rows = 2
# # columns = 3
# # axes = []
# # fig = plt.figure()
# # for i in range(rows*columns):
# #     subplot_title=(l1[i])
# #     axes.append(fig.add_subplot(rows,columns,i+1))
# #     axes[-1].set_title(subplot_title)
# #     plt.imshow(img_l[i],cmap='gray')
# # fig.tight_layout()
# # plt.show()
#
# #cv2.imshow('image',img2)
# #cv2.waitKey(0)
