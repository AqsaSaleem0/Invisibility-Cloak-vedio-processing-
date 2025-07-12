import cv2
import numpy as np

def dynamic_crop(img, padding=20):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the olive green cloak (based on your values)
    lower_bound = np.array([9, 19, 0])        # Lower Hue, Sat, Val
    upper_bound = np.array([60, 80, 170])  # Upper Hue, Sat, Val
    
    # Create a mask for the cloak
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to remove noise and improve mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box coordinates for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding to the bounding box
        x_pad = max(x - padding, 0)
        y_pad = max(y - padding, 0)
        w_pad = min(x + w + padding, img.shape[1]) - x_pad
        h_pad = min(y + h + padding, img.shape[0]) - y_pad


        # Crop the image around the padded bounding box
        cropped_img = img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

    else:
        print("No contours detected. Adjust the HSV range or check the mask.")

    return cropped_img


for i in range(543):

    img=cv2.imread(f"dataset/1_raw_frames/frame_{i}.jpg")
    croped_image=dynamic_crop(img, padding=150)
    cv2.imwrite(f"dataset/2_preprocessed_frames/croped_frames/frame_{i}.jpg",croped_image)


