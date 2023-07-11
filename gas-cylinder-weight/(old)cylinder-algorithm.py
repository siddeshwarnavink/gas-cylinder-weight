from sys import exit

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract

img= cv2.imread('./images/Cylinder_image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

_, binary_image = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(img_gray)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
image = cv2.bitwise_and(img, img, mask=mask)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Set filtering parameters
min_area_threshold = 100  # Minimum contour area threshold
aspect_ratio_min = 1  # Minimum aspect ratio
aspect_ratio_max = 1.2  # Maximum aspect ratio
min_height = 200  # Minimum contour height

coordinates = []

# Iterate over contours
for contour in contours:
    # Compute contour area and bounding rectangle
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    # Filter contours based on area, aspect ratio, height, and color
    if (
        area > min_area_threshold
        and aspect_ratio_min < w / h < aspect_ratio_max
        and h > min_height and h <= 300
    ):
        cropped_image = image[y:y+h, x:x+w]

        # Extract the region of interest containing the number
        number_region = image[y:y + h, x:x + w]

        # Convert the number region to grayscale
        gray_number = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)

        # Apply OCR using Tesseract
        number = pytesseract.image_to_string(gray_number)

        # Check if the extracted text contains only digits
        if number.isdigit():
            # Print the detected number and its location
            print("Detected number:", number)
            print("Location (x, y):", x, y)

        # Draw a rectangle around the detected number
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        coordinates.append((x, y))


# Write the image with the detected numbers to a file
cv2.imwrite('./images/output_image.jpg', image)

print("Output image saved successfully.")

x = coordinates[0][0]
y = coordinates[0][1]

angle = 0
width = 500
height = 250
# Get the image dimensions
height2, width2 = image.shape[:2]
numbers = []
while (angle < 360):
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width2/2, height2/2), angle, 1)

    # Perform rotation using warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width2, height2))
    cropped_image = rotated_image[y:y+height, x:x+width]
    # cv2.imshow('test',cropped_image)
    # cv2.waitKey(0)
    recognized_text = pytesseract.image_to_string(cropped_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    if(recognized_text != None):
        recognized_text = str(recognized_text).strip()
        if(recognized_text != ''):
            numbers.append(int(recognized_text.strip()))
    angle+=10
print(numbers)

# cropped_numbers = [num for num in numbers if 100 <= num <= 200]
# print(cropped_numbers)

# max_number = max(cropped_numbers)
# final_weight = max_number / 10
# print(final_weight)