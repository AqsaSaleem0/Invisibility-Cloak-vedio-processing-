import cv2
import numpy as np
import subprocess
import os


video_path = 'video5.mp4' 
detect_script = '/home/aqsa/Invisibility_clock_project/yolov5/detect.py'  
model_path = '/home/aqsa/Invisibility_clock_project/yolov5/runs/train/exp/weights/best.pt'
output_dir = 'output'  
output_video_path = 'output_video2.mp4' 


cap = cv2.VideoCapture(video_path)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


background_frames = []
num_background_frames = 30  

print("Capturing background...")
for i in range(num_background_frames):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)


background = np.mean(background_frames, axis=0).astype(np.uint8)


frame_index = 0
print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

  
    frame_path = f'frame_{frame_index}.jpg'
    cv2.imwrite(frame_path, frame)

 
    detection_output = f'{output_dir}/frame_{frame_index}.jpg'
    command = [
        'python3', detect_script,
        '--weights', model_path,
        '--source', frame_path,
        '--save-txt', '--project', output_dir, '--name', 'detections', '--exist-ok'
    ]
    subprocess.run(command, check=True)

    
    detection_label_path = os.path.join(output_dir, 'detections', 'labels', f'frame_{frame_index}.txt')
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if os.path.exists(detection_label_path):
        with open(detection_label_path, 'r') as f:
            for line in f:
                _, x_center, y_center, width, height = map(float, line.strip().split())
                h, w = frame.shape[:2]
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

    
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    result = cv2.add(cloak_area, non_cloak_area)

    out.write(result)

    cv2.imshow("Invisibility Cloak Effect", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    os.remove(frame_path)
    frame_index += 1


cap.release()
out.release()  
cv2.destroyAllWindows()
