'''
Solution #2
Using a algorithm to detect the numbers. And process them with OCR
'''
import sys
import re
import time

import cv2
import numpy as np
import pytesseract

from shared import getMonochromeImage

if sys.platform=='win32':
    pytesseract.pytesseract.tesseract_cmd=r'C:/Program Files/Tesseract-OCR/tesseract.exe'

startTime=time.time()

def getWeightPrediction(images):
    predictions=[]

    def sortedKeysToNumber(keys):
        if len(keys)!=3: return -1.0
        number=float(f'{keys[0]}{keys[1]}.{keys[2]}')
        return number

    for img in images:
        test_array=[1,3,5]
        location_dictionary={}
        for number in test_array :
            template=cv2.imread(f'./images/template/{number}.jpg')
            method=cv2.TM_CCOEFF_NORMED
            result=cv2.matchTemplate(img,template,method)
            threshold=0.6
            best_match=np.max(result)
            if best_match>=threshold:
                min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(result)
                top_left=max_loc
                x_location=top_left[0]
                location_dictionary[f'{number}']=x_location
        sorted_keys = [k for k, v in sorted(location_dictionary.items(), key=lambda item: item[1])]
        predictions.append(sortedKeysToNumber(sorted_keys))
    return predictions

def getInitialCroppedImage(image):
    image_height,image_width,_=image.shape
    crop_width=1400
    crop_height=1400

    x1=int(image_width/2-crop_width/2)
    y1=int(image_height/2-crop_height/2)
    x2=int(image_width/2+crop_width/2)
    y2=int(image_height/2+crop_height/2)

    cropped_image=image[y1:y2,x1:x2]
    return cropped_image

def getContours(image):
    thresh=cv2.adaptiveThreshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 11, 7)
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours

loadedImage=cv2.imread('./images/Cylinder_image.jpg')
image=getMonochromeImage(loadedImage)
cropped_image=getInitialCroppedImage(image)
originalImage=cropped_image.copy()

def getNumberPos(image):
    contours=getContours(image)
    number_positions=[]

    for contour in contours:
        area=cv2.contourArea(contour)
        x,y,w,h=cv2.boundingRect(contour)

        if (
            area>1000
            and 1<w/h<1.2
            and h>200 and h<=300
        ):
            number_positions.append({'pos':(x,y),'cnt':contour})
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return number_positions

def getNearbyContours(image,posList,contours):
    max_distance=200
    filtered_contours=[]

    for cntData in posList:
        point_of_interest=cntData['pos']
        for cnt in contours:
            area=cv2.contourArea(cnt)
            x,y,w,h=cv2.boundingRect(cnt)

            if area>1000:
                M=cv2.moments(cnt)
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                distance=np.sqrt((cx-point_of_interest[0])**2+(cy-point_of_interest[1])**2)

                if distance<=max_distance:
                    filtered_contours.append({'pos':(x,y),'cnt': cnt})

    return filtered_contours

number_positions=getNumberPos(cropped_image)
nearby_contours=getNearbyContours(cropped_image,number_positions,getContours(cropped_image))

def filterNumberImages(img,number_positions,nearby_contours):
    img_counter=0
    extract_images=[]

    def findAngle(x,y,w,h):
        rect_center_x=x+w//2
        rect_center_y=y+h//2
        img_center_x=img.shape[1]//2
        img_center_y=img.shape[0]//2
        ref_point_x=img_center_x
        ref_point_y=rect_center_y
        distance_yellow_to_orange=np.sqrt((ref_point_x-img_center_x)**2+(ref_point_y-img_center_y)**2)
        distance_red_to_orange=np.sqrt((ref_point_x-rect_center_x)**2+(ref_point_y-rect_center_y)**2)
        radians=np.arctan(distance_red_to_orange/distance_yellow_to_orange)
        angle=np.degrees(radians)
        return angle
    
    def check_contour_overlap(contour1, contour2):
        x1,y1,w1,h1=cv2.boundingRect(contour1)
        x2,y2,w2,h2=cv2.boundingRect(contour2)

        rect1_tl=(x1,y1)
        rect1_br=(x1+w1,y1+h1)
        rect2_tl=(x2,y2)
        rect2_br=(x2+w2,y2+h2)

        if rect1_tl[0]<rect2_br[0]and rect1_br[0]>rect2_tl[0]and rect1_tl[1]<rect2_br[1]and rect1_br[1]>rect2_tl[1]:
            return True
        else:
            return False

    for parentCntData in nearby_contours:
        for childCntData in number_positions:
            if(check_contour_overlap(parentCntData['cnt'],childCntData['cnt'])):
                img_counter+=1
                x,y,w,h=cv2.boundingRect(parentCntData['cnt'])
                if x<800 and y>800:
                    global originalImage
                    angle=findAngle(x,y,w,h)
                    rows,cols=cropped_image.shape[:2]
                    M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
                    originalImage=cv2.warpAffine(originalImage,M,(cols,rows))
                    break
                extract_image=originalImage[y:y+h,x:x+w]
                extract_images.append(extract_image)
    return extract_images

extract_images=filterNumberImages(cropped_image,number_positions,nearby_contours)

def getRotationImages(img):
    angle=0
    cropped_number_image=[]
    for _ in range(0,3):
        rows,cols=img.shape[:2]
        M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        rotated_image_copy=cv2.warpAffine(img,M,(cols,rows))
        angle+=110
        cropped_number=rotated_image_copy[1000:1350,450:900]
        cropped_number_image.append(cropped_number)
    return cropped_number_image

cropped_number_image=getRotationImages(originalImage)

def preditNumberPredictions(images_list):
    numbers_predicted=[]

    def remove_special_characters_and_letters(input_string):
        cleaned_string=re.sub('[^0-9]','',input_string)
        return cleaned_string

    for img in images_list:
        resize_img=cv2.resize(img,(250,250))
        img_array=np.array(cv2.cvtColor(resize_img,cv2.COLOR_RGB2GRAY))
        for rows in range(len(img_array[0])):
            for cols in range(len(img_array)):
                if img_array[rows][cols]<100:
                    img_array[rows][cols]=255
                else:
                    img_array[rows][cols]=0
        white_img=cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        recognized_text=pytesseract.image_to_string(white_img,config='--psm 11 --oem 3 ')
        number=remove_special_characters_and_letters(recognized_text)
        if number==''or float(number)<20:
            actual_number='Error'
        elif float(number)<100:
            actual_number=float(number)/10+10

        else:
            actual_number=float(number)/10
        numbers_predicted.append(actual_number)
    return numbers_predicted

# number_predictions=preditNumberPredictions(cropped_number_image)
# print(number_predictions)
weight_predictions=getWeightPrediction(cropped_number_image)
print(weight_predictions)
endTime=time.time()
timeTaken=round(((endTime-startTime)*10**3)/1000,2)
print(f'Elapsed: {timeTaken}s')