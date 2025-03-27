### Step 2: Define the main pipeline with action recognition, UI overlay, and logging
# main.py
import cv2
import numpy as np
import datetime
from ultralytics import YOLO
from utils import draw_keypoints, draw_bounding_boxes, assign_ids, preprocess_keypoints, draw_ui_overlay, log_action
from model import load_action_model, predict_action_sequence
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load YOLO model
yolo = YOLO('yolov8n.pt')

# Load the pre-trained action recognition model
action_model = load_action_model('action_recognition_model.h5')

# Tracking people and their pose history
tracked_people = {}
pose_sequences = {}  # Store sequences of poses for each person
action_stats = {}  # Store live action counts
next_id = 0
sequence_length = 30  # Number of frames to store for each person

# Initialize log file
log_file = 'action_log.txt'
with open(log_file, 'w') as f:
    f.write('Timestamp, Person ID, Action, Confidence (%)\n')

# Video capture (use your own video file)
video_path = r'C:\Users\second\OneDrive\Desktop\65pro\static\uploads\WhatsApp Video 2025-03-11 at 21.56.14_864b3342.mp4'  # Update this to your file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO for human detection
    results = yolo.predict(frame)
    detections = []
    for result in results:
        if hasattr(result, 'boxes'):
            detections.extend(result.boxes.data.cpu().numpy())
    
    # Assign IDs to detected persons
    tracked_people = assign_ids(detections, tracked_people, next_id)
    next_id = max(tracked_people.keys(), default=0) + 1

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process each detected person
    for person_id, (x1, y1, x2, y2) in tracked_people.items():
        person_frame = frame[int(y1):int(y2), int(x1):int(x2)]
        rgb_person = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_person)

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append(((landmark.x * (x2 - x1)) + x1, (landmark.y * (y2 - y1)) + y1))

        # Preprocess keypoints and store sequences
        keypoints = preprocess_keypoints(keypoints)
        if person_id not in pose_sequences:
            pose_sequences[person_id] = deque(maxlen=sequence_length)
        pose_sequences[person_id].append(keypoints)

        # Predict action if sequence is long enough
        action, confidence = 'Collecting data...', 0.0
        if len(pose_sequences[person_id]) == sequence_length:
            action, confidence = predict_action_sequence(action_model, np.array(pose_sequences[person_id]))
            action_stats[person_id] = {'action': action, 'confidence': round(confidence * 100, 2)}
            log_action(log_file, person_id, action, confidence)

        # Draw bounding boxes and actions
        draw_bounding_boxes(frame, [[x1, y1, x2, y2, 1, 0]])
        draw_keypoints(frame, keypoints)
        cv2.putText(frame, f'ID {person_id}: {action} ({int(confidence * 100)}%)', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw UI overlay with live action stats
    draw_ui_overlay(frame, action_stats)

    # Display frame
    cv2.imshow('Human Action Recognition with AI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
