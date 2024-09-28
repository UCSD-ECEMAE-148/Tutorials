import cv2 as cv2 #importing the library
import numpy as np

def show_image(name,img): #function for displaying the image
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    # show_image('Canny',img_canny)
    return img_canny

def region_of_interest(image): #function for extracting region of interest
    #bounds in (x,y) format
    # bounds = np.array([[[140,539],[425,330],[530,330],[860,539]]],dtype=np.int32)

    bounds = np.array([[[300,539],[300,0],[750,0],[750,539]]],dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,0,0])
    masked_image = cv2.bitwise_and(image,mask)
    # show_image('mask',mask) 
    return masked_image


def draw_lines(img,lines): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

def get_coordinates(img,line_parameters): #functions for getting final coordinates
    slope=line_parameters[0]
    intercept = line_parameters[1]
    # y1 =300
    # y2 = 120
    y1=img.shape[0]
    y2 = 0.4*img.shape[0]
    x1= int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,int(y1),x2,int(y2)]

def compute_average_lines(img,lines):
    left_lane_lines=[]
    right_lane_lines=[]
    left_weights=[]
    right_weights=[]
    for points in lines:
        x1,y1,x2,y2 = points[0]
        if x2==x1:
            continue     
        parameters = np.polyfit((x1,x2),(y1,y2),1) #implementing polyfit to identify slope and intercept
        slope,intercept = parameters
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slope <0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)         
        else:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)
    #Computing average slope and intercept
    left_average_line = np.average(left_lane_lines,axis=0)
    right_average_line = np.average(right_lane_lines,axis=0)
    print(left_average_line,right_average_line)
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img,left_average_line)
    right_fit_points = get_coordinates(img,right_average_line) 
    print(left_fit_points,right_fit_points)
    return [[left_fit_points],[right_fit_points]] #returning the final coordinates


###*****************************************************************###
##Color Segmentation followed by Perspective Transformation:
image2 = cv2.imread('solidWhiteRight.jpg')
show_image('input',image2)
#Selection of Region of Interest for perspective Transformation
bounds = np.array([[[140,539],[425,330],[530,330],[860,539]]],dtype=np.float32)
#Required boundary of the transformed image
new_perspective = np.array([[[300,539],[300,0],[750,0],[750,539]]],dtype=np.float32)
#Computing the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(bounds,new_perspective)
#Applying the transformation matrix on the input image for perspective transformation
lane_warped_image = cv2.warpPerspective(image2,transformation_matrix,image2.shape[:2][::-1],flags=cv2.INTER_LINEAR)
show_image('warped',lane_warped_image)
lane_image_2 = np.copy(lane_warped_image)
#Converting modified image to HLS space
lane_image_2 =cv2.cvtColor(lane_image_2,cv2.COLOR_BGR2HLS)
show_image('hls_input',lane_image_2)
#Defining upper and lower boundaries for white color
lower_white_hls = np.uint8([  0, 220,   0])
upper_white_hls = np.uint8([255, 255, 255])
#Using bitwise operators to segment out white colors
lane_white_mask = cv2.inRange(lane_image_2,lower_white_hls,upper_white_hls)
show_image('whitemask',lane_white_mask)
lane_image_mask = cv2.bitwise_and(lane_image_2,lane_image_2,mask=lane_white_mask)
# show_image('bitmask',lane_image_mask)
#Implementing Canny and Hough
lane_canny_2 = find_canny(lane_image_mask,50,200)
lane_roi_2 = region_of_interest(lane_canny_2)
show_image('canny',lane_roi_2)
lane_lines_2 = cv2.HoughLinesP(lane_roi_2,1,np.pi/180,5,5,15)
lane_lines_plotted_2 = draw_lines(lane_image_2,lane_lines_2)
show_image('lines',lane_lines_plotted_2)
result_lines_2 = compute_average_lines(lane_image_2,lane_lines_2)
final_lines_mask_2 = draw_lines(lane_image_2,result_lines_2)
show_image('final',final_lines_mask_2)

#Plotting the final lines on main image
for points in result_lines_2:
    x1,y1,x2,y2 = points[0]
    cv2.line(lane_image_2,(x1,y1),(x2,y2),(0,0,255),2)

show_image('outupt',lane_image_2)
# ###******************************************************************************#####
