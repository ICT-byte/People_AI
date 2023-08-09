import cv2
import torch
import numpy as np
import time

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
COCO_INSTANCE_CATEGORY_NAMES = model.names

cap = cv2.VideoCapture(0)

last_detected_time = time.time()
timeout = 0.1 * 60  # 15 minutes

while True:
    _, frame = cap.read()

    with torch.no_grad():
        results = model(frame, size=640)

    person_detected = False
    for *xyxy, conf, label in results.xyxy[0]:
        if COCO_INSTANCE_CATEGORY_NAMES[int(label)] == "person" and conf > 0.5:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_detected = True

    cv2.imshow("Object Detection", frame)

    current_time = time.time()
    if person_detected:
        print(1)
        last_detected_time = current_time
    elif current_time - last_detected_time > timeout:
        print(0)
        last_detected_time = current_time

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
