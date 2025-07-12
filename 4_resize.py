import cv2

for i in range(543):
    img=cv2.imread(f"dataset/2_preprocessed_frames/croped_frames/frame_{i}.jpg")
    resized_image=cv2.resize(img, (640,640))
    cv2.imwrite(f"dataset/2_preprocessed_frames/resized_frames/frame_{i}.jpg",resized_image)
    



