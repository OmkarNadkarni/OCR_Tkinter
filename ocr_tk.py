from PIL import ImageTk
from tkinter import filedialog
from tkinter import *
from PIL import Image
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from ocr_functions import *

'''
def grayscale(img):
def blur(img,KERNEL_SIZE,GAUSSIAN=True):
def thresholding(img, isAdaptive,TH=70):
def noise_rem(img):
def erosion(img,kernel,no_iter):
def dilation(img,kernel,no_iter):
def returnText(img):
def imgBbox(img):
def ImageResize(path):
'''
def openfile():
    root.filename = filedialog.askopenfilename(title="select a image",filetypes=(("jpg files","*.jpg"),("all files","*.* ")))
    img = cv2.imread(root.filename)
    img = cv2.selectROI(img)
    global file_label
    file_label.config(text=root.filename)
def getOCR():
    global my_img
    img = cv2.imread(root.filename)
    img = grayscale(img)
    img = blur(img,3)
    img = thresholding(img,1)
    img = noise_rem(img)
    text = returnText(img)
    print(text)
    width,height = img.shape
    x,y = min(width,600),min(height,600)
    img = cv2.resize(img,(x,y))
    cv2.imwrite('D:/OCR/images/temp.jpg',img)
    my_img = ImageTk.PhotoImage(Image.open(r'D:\OCR\images\temp.jpg').convert('LA'))
    my_label = Label(image=my_img).grid(row=2,column=0)
    result_label = Label(text=text).grid(row=3,column=0)

root = Tk()
root.title('OCR')
root.geometry('800x800')
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 6'

file_label = Label(text='sample')
file_label.grid(row=0,column=1,columnspan=2)
browse_button= Button(text='browse file',command=openfile,padx=20).grid(row=0,column=0)
ocr_button = Button(text="OCR",command=getOCR,padx=35).grid(row=1,column=0)

my_label = Label()

root.mainloop()
