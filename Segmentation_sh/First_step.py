#Импорт всех необходимых компонентов и утилит
import numpy as np
import cv2
import os
from cv2 import createCLAHE
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature,morphology
#from google.colab.patches import cv2_imshow

base_path = "Segmentation_sh"
test_folder = "test_data"
blue_tr_filename="2021-10-20_13_52_19_237.png"
# filename='783.png'
filename = blue_tr_filename
path = os.path.join(test_folder, filename)
# path = "/".join((base_path,test_folder, filename))
# path = "C:\\Users\\zgstv\\JupyterLab Notebooks\\Mag-Project\\Segmentation_sh\\test_data\\2021-10-20_13_52_19_237.png"
img = cv2.imread(path)
a= filename[0:len(filename) - 4]
print(os.access(path,os.R_OK))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fn=a+'_A.png'
# img_gray,sqsumm,titled= cv2.integral3(img_gray/255)
# cv2.imwrite(fn,img_gray)

gradX = cv2.Sobel(img_gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(img_gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
#вычитание
img_B = cv2.subtract(gradX, gradY)
#gradient=np.sqrt(gradX**2+gradY**2)
#gradient = cv2.convertScaleAbs(gradient)
# threshold the image
(_, img_B) = cv2.threshold(img_B, 200, 255, cv2.THRESH_BINARY)
fn=a+'_B.png'
cv2.imwrite(fn,img_B)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(16,2))
img_C = cv2.morphologyEx(img_B, cv2.MORPH_CLOSE, kernel)
img_C = cv2.erode(img_C, None, iterations = 4)
img_C = cv2.dilate(img_C, None, iterations = 4)
fn=a+'_C.png'
cv2.imwrite(fn,img_C)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
img_C_ = cv2.morphologyEx(img_C, cv2.MORPH_OPEN,kernel)
img_C_ = cv2.erode(img_C_, None, iterations = 4)
img_C_ = cv2.dilate(img_C_, None, iterations = 4)
#(x,y),(MA,ma),angle = cv2.fitEllipse(img_C)# Ориентация объекта

fn=a+'_C_.png'
cv2.imwrite(fn,img_C_)

#img_B = cv2.imread(filename,0)
contours_C, hierarchy = cv2.findContours(img_C.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #create an empty image for contours img_contours = np.zeros(img.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save_results image cv2.imwrite('D:/contours.png',img_contours)
img_contours_C = np.zeros(img_C.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save_results image cv2.imwrite('D:/contours.png',img_contours)
cv2.drawContours(img_contours_C,contours_C,-1,(255,0,0),-1)
cv2.imwrite('contours_C.png',img_contours_C)



