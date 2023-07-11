'''
Solution #1
Using a rotating cursor find out the number. 
'''
import os

import cv2
import pytesseract

import shared

loadedImage=cv2.imread('./images/Cylinder_image.jpg')
image=shared.getMonochromeImage(img=loadedImage)

'''
Cursor
'''
CURSOR_X=900
CURSOR_Y=1500
CURSOR_WIDTH=500
CURSOR_HEIGHT=300
CURSOR_ROTATE_BY=5

def cursorRotateImg(c,img,angle):
    imgHeight,imgWidth=img.shape[:2]
    rotation_matrix=cv2.getRotationMatrix2D((imgWidth/2,imgHeight/2),angle,1)
    rotated_image=cv2.warpAffine(img,rotation_matrix,(imgWidth,imgHeight))
    cropped_image=rotated_image[c['yPos']:c['yPos']+c['height'],c['xPos']:c['xPos']+c['width']]
    return cropped_image

def recognizedNumberFromImg(img):
    if os.name=='nt':
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    recognized_text=pytesseract.image_to_string(img,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    if(recognized_text!=None):
        recognized_text=str(recognized_text).strip()
        if(recognized_text!=''):
            return int(recognized_text)
    return None

cursor={
    'xPos':CURSOR_X,
    'yPos':CURSOR_Y,
    'width':CURSOR_WIDTH,
    'height':CURSOR_HEIGHT
}
angle=0
numbers=[]

while(angle<360):
    cursorImg=cursorRotateImg(cursor,image,angle)
    # cv2.imshow('Cursor Preview',cursorImg)
    # cv2.waitKey(0)
    numFromImg=recognizedNumberFromImg(cursorImg)
    if(numFromImg):
        numbers.append(numFromImg)
    angle+=CURSOR_ROTATE_BY
cv2.destroyAllWindows()
cropped_numbers=[num for num in numbers if 100 <= num <= 200]
avg_number=sum(cropped_numbers) / len(cropped_numbers)
final_weight=avg_number/10
print(f'Weight: {final_weight}kg')