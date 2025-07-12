import cv2
import imgaug.augmenters as iaa
import imgaug.augmentables.bbs as ia_bbs
import numpy as np


# Define different augmentation pipelines
augmenters_list = [
    iaa.Sequential([iaa.Fliplr(0.5)]),  # Horizontal flip
    iaa.Sequential([iaa.Affine(rotate=(-30, 30))]),  # Rotate between -30 and 30 degrees
    iaa.Sequential([iaa.Multiply((0.8, 1.2))]),  # Adjust brightness
    iaa.Sequential([iaa.Affine(scale=(0.8, 1.2))]),  # Scale images
    iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-30, 30))]),  # Flip and Rotate
]
count=0
# Paths
input_dir = "dataset/2_preprocessed_frames/labeled_frames"
output_dir = "dataset/2_preprocessed_frames/augmented_frames"

# Process all images
for i in range(542):  # Process all frames
    image_path = f"{input_dir}/frames/frame_{i}_labeled.jpg"
    label_path = f"{input_dir}/annotations/frame_{i}.txt"

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {i} not found. Skipping.")
        continue

    # Read YOLO annotations
    with open(label_path, "r") as f:
        annotations = f.readlines()

    bbs = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        img_h, img_w = image.shape[:2]

        # Convert YOLO to absolute coordinates
        x1 = (x_center - width / 2) * img_w
        y1 = (y_center - height / 2) * img_h
        x2 = (x_center + width / 2) * img_w
        y2 = (y_center + height / 2) * img_h

        bbs.append(ia_bbs.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))

    # Convert to imgaug format
    bbs_on_image = ia_bbs.BoundingBoxesOnImage(bbs, shape=image.shape)

    # Apply different augmentation pipelines
    for j, augmenter in enumerate(augmenters_list):  # Iterate over different augmentations
        count=count+1
        augmented_image, augmented_bbs = augmenter(image=image, bounding_boxes=bbs_on_image)

        # Save augmented image
        output_image_path = f"{output_dir}/frames/frame_{i}_aug_{j}.jpg"
        cv2.imwrite(output_image_path, augmented_image)

        # Save augmented YOLO annotations
        img_h, img_w = augmented_image.shape[:2]
        output_label_path = f"{output_dir}/annotations/frame_{i}_aug_{j}.txt"
        with open(output_label_path, "w") as f:
            for bb in augmented_bbs.bounding_boxes:
                # Convert back to YOLO format
                x_center = (bb.x1 + bb.x2) / 2 / img_w
                y_center = (bb.y1 + bb.y2) / 2 / img_h
                width = (bb.x2 - bb.x1) / img_w
                height = (bb.y2 - bb.y1) / img_h

        # Clip the values to be within the range [0, 1]
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)  # Corrected line
                height = np.clip(height, 0, 1)  # Corrected line
                f.write(f"{bb.label} {x_center} {y_center} {width} {height}\n")
print("total frames:", count)
print("Augmentation completed.")
