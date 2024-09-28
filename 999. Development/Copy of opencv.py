import cv2 as cv2 #importing the library
img1= cv2.imread('geisel.jpg') #reading from the image file "geisel.jpg" located in the same folder as python file
print(img1[0][0])
# print(type(img1))
# cv2.imshow('library',img1) #displaying the read image
# cv2.waitKey(0) 
# cv2.destroyAllWindows() #destroying the windows


import numpy as np #importing numpy library

geisel_low= np.copy(img1)
geisel_low = cv2.resize(geisel_low,(450,400)) #Resizing the image with width 450 pixels and height 400 pixels
print(geisel_low[0][0])
# cv2.imshow('geisel_low',geisel_low) #displaying the read image
# cv2.waitKey(0) 
# cv2.destroyAllWindows() #destroying the windows


geisel_low = cv2.cvtColor(geisel_low
,cv2.COLOR_BGR2GRAY)
print(geisel_low)
# cv2.imshow('check',geisel_low)
# geisel_gray = cv2.cvtColor(geisel_low,cv2.COLOR_RGB2HSV) # converting from BGR2GRAY
# print(geisel_gray[0][0])
# cv2.imshow('geisel_gray',geisel_low)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

geisel_blur = cv2.GaussianBlur(geisel_low,(25,25),0) # Applying Gaussian Blur over Grayscal image
# cv2.imshow('geisel_blur',geisel_blur)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# sobel_geisel = cv2.Sobel(geisel_blur,cv2.CV_64F,1,1,3) #Applying the sobel filter
# cv2.imshow('sobel',sobel_geisel)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

canny_geisel = cv2.Canny(geisel_blur,100,200) #Applying Canny Edge Detector with 100-200
cv2.imshow('canny',canny_geisel)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# hough_geisel = cv2.HoughLinesP(canny_geisel,1,np.pi/180,15,5,10) #Applying Hough lines detection
# print(len(hough_geisel))

# for points in hough_geisel:
#       # Extracted points nested in the list
#     x1,y1,x2,y2=points[0]
#     # Draw the lines joing the points
#     # On the original image
#     cv2.line(geisel_low,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imshow('hough',geisel_low)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()