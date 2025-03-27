### Step 4: Utilities
# utils.py
import cv2
import numpy as np
import datetime
from scipy.spatial import distance

def draw_keypoints(frame, keypoints):
    if not isinstance(keypoints, (list, np.ndarray)) or len(keypoints) == 0:
        return  # Skip if keypoints are invalid

    keypoints = np.array(keypoints).reshape(-1, 2)  # Ensure keypoints are pairs of (x, y)

    for x, y in keypoints:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

def draw_bounding_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def assign_ids(detections, tracked_people, next_id):
    new_tracked = {}
    detection_centers = [(int((x1 + x2) // 2), int((y1 + y2) // 2)) for x1, y1, x2, y2, _, _ in detections]

    for person_id, (x1, y1, x2, y2) in tracked_people.items():
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distances = [distance.euclidean(person_center, det_center) for det_center in detection_centers]
        if distances and min(distances) < 50:
            idx = distances.index(min(distances))
            new_tracked[person_id] = detections[idx][:4]
            detection_centers.pop(idx)
            detections.pop(idx)
    
    for det in detections:
        new_tracked[next_id] = det[:4]
        next_id += 1

    return new_tracked

def preprocess_keypoints(keypoints):
    keypoints = np.array(keypoints)
    if keypoints.shape != (17, 2):
        keypoints = np.zeros((17, 2))
    keypoints = keypoints.flatten() / 1000
    return keypoints

def draw_ui_overlay(frame, action_stats):
    y_offset = 30
    for person_id, stats in action_stats.items():
        cv2.putText(frame, f"ID {person_id}: {stats['action']} ({stats['confidence']}%)", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40

def log_action(log_file, person_id, action, confidence):
    with open(log_file, 'a') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{timestamp}, {person_id}, {action}, {round(confidence * 100, 2)}\n')
