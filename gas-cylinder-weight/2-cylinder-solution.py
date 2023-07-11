'''
Solution #2
Using a algorithm to detect the numbers and process is using ML
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract

def getWhiteTextImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    _, binary_image = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def getNumbersFromImage(image):
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_counter = 0
    number_images = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if (
            area > 100
            and 1 < w / h < 1.2
            and h > 200 and h <= 300
        ):
            img_counter += 1
            w*=2
            h*=2
            cropped_image = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(f'./images/output_images/{img_counter}.jpg', cropped_image)
            number_images.append(cropped_image)

    cv2.imwrite('./images/output_images/final_image.jpg', image)
    return number_images

loadedImage = cv2.imread('./images/Cylinder_image.jpg')
image = getWhiteTextImage(loadedImage)
numberImages = getNumbersFromImage(image)

'''
for numberImage in numberImages:
    angle = 0
    numbers = []
    while angle < 360:
        width = 500
        height = 250
        height2, width2 = image.shape[:2]
        # rotate image
        rotation_matrix = cv2.getRotationMatrix2D((width2/2, height2/2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width2, height2))

        recognized_text = pytesseract.image_to_string(rotated_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        if(recognized_text != None):
            recognized_text = str(recognized_text).strip()
            if(recognized_text != ''):
                numbers.append(int(recognized_text.strip()))
        angle += 5
    print(numbers)
'''