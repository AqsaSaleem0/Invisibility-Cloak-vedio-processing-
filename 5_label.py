import cv2
import numpy as np

def detect_and_label_rotated(img):
    """
    Detects the cloak in the image, labels it with a rotated bounding box,
    and calculates YOLO-format annotation.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for the cloak
    lower_bound = np.array([9, 19, 0])
    upper_bound = np.array([60, 80, 170])

    # Create mask and clean it
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the rotated bounding box
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)  # Ensure integer coordinates

        # Draw the rotated bounding box on the image
        labeled_img = img.copy()
        cv2.drawContours(labeled_img, [box], 0, (0, 255, 0), 2)

        # Calculate YOLO format
        x, y, w, h = cv2.boundingRect(largest_contour)
        img_h, img_w = img.shape[:2]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        bbox_width = w / img_w
        bbox_height = h / img_h
        # Clip the values to be within the range [0, 1]
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        bbox_width = np.clip(bbox_width, 0, 1)
        bbox_height = np.clip(bbox_height, 0, 1)

        return labeled_img, (0, x_center, y_center, bbox_width, bbox_height)
    else:
        print("No contours detected.")
        return img, None

# Process all 542 frames
for i in range(542):
    # Load the image
    img = cv2.imread(f"dataset/2_preprocessed_frames/resized_frames/frame_{i}.jpg")
    if img is None:
        print(f"Frame {i} not found. Skipping.")
        continue

    # Detect and label the image
    labeled_img, yolo_bbox = detect_and_label_rotated(img)

    # Save the labeled image
    output_img_path = f"dataset/2_preprocessed_frames/labeled_frames/frames/frame_{i}_labeled.jpg"
    cv2.imwrite(output_img_path, labeled_img)

    # Save YOLO annotations
    if yolo_bbox:
        with open(f"dataset/2_preprocessed_frames/labeled_frames/annotations/frame_{i}.txt", "w") as f:
            f.write(" ".join(map(str, yolo_bbox)))


print("All frames processed and labeled.")
