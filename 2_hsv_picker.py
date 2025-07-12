import cv2
import numpy as np

def display_hsv_picker(image_path):
    def nothing(x):
        pass

    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (600, 400))  # Resize for easier handling
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a window
    cv2.namedWindow("HSV Picker")

    # Create trackbars for hue, saturation, and value
    cv2.createTrackbar("Lower Hue", "HSV Picker", 0, 179, nothing)
    cv2.createTrackbar("Upper Hue", "HSV Picker", 179, 179, nothing)
    cv2.createTrackbar("Lower Sat", "HSV Picker", 0, 255, nothing)
    cv2.createTrackbar("Upper Sat", "HSV Picker", 255, 255, nothing)
    cv2.createTrackbar("Lower Val", "HSV Picker", 0, 255, nothing)
    cv2.createTrackbar("Upper Val", "HSV Picker", 255, 255, nothing)

    while True:
        # Get current trackbar positions
        lh = cv2.getTrackbarPos("Lower Hue", "HSV Picker")
        uh = cv2.getTrackbarPos("Upper Hue", "HSV Picker")
        ls = cv2.getTrackbarPos("Lower Sat", "HSV Picker")
        us = cv2.getTrackbarPos("Upper Sat", "HSV Picker")
        lv = cv2.getTrackbarPos("Lower Val", "HSV Picker")
        uv = cv2.getTrackbarPos("Upper Val", "HSV Picker")

        # Create a mask with the current HSV range
        lower_bound = np.array([lh, ls, lv])
        upper_bound = np.array([uh, us, uv])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Display the result
        result = cv2.bitwise_and(img, img, mask=mask)

        # Show images
        # cv2.imshow("Original Image", img)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Filtered Image", result)

        # Break loop on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# Usage
display_hsv_picker("dataset/3_preprocessed_frames/resized_frames/frame_0.jpg")
