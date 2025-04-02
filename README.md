# Human Action Recognition with YOLO and TensorFlow

This project implements **Real-Time Human Action Recognition** using **YOLOv8** for person detection and **TensorFlow/Keras** for action classification. It tracks multiple people, identifies their actions, and displays real-time stats.

---

## 🚀 **Features**
- **Real-time human detection** with YOLOv8.
- **Pose estimation** using MediaPipe.
- **Action recognition** with a pre-trained TensorFlow model.
- **Multi-person tracking** with unique IDs.
- **Live statistics** and action logs.
- **Video display with bounding boxes and action labels**.

---

## 🛠️ **Tech Stack**
- Python 3.x
- OpenCV
- YOLOv8 (Ultralytics)
- TensorFlow/Keras
- MediaPipe
- NumPy
- Matplotlib

---
## Run the main program:
- python main.py

---
## Model Training and Testing:
**To view the model architecture:**
- python train_action_model.py


## Configuration:
## **Modify the video_path in main.py to use your own video file**
- video_path = r'path_to_your_video_file.mp4'

## 📊 Logs and Output:
 **Logs are saved in action_log.txt with:**
- Timestamp, Person ID, Action, Confidence (%)

- Real-time stats are displayed on the video frame.
