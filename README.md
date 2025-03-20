# Traffic Sign Detection

## Overview
This project implements a **Traffic Sign Detection** system using **YOLO** for object detection. The model is capable of detecting and classifying **45 different traffic signs** in real-time from video streams or images.

## Features
- **Real-time traffic sign detection** using YOLO.
- **Supports video input and webcam feeds**.
- **Classifies traffic signs into 45 categories**.
- **Outputs processed video with detected signs**.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mona1Tarek/traffic-sign-detection.git
   cd traffic-sign-detection
   ```
2. Install required dependencies:
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib
   ```

## Files in the Repository
- `best.pt`: Pre-trained YOLO model weights.
- `trafficdetection_images_ultra.py`: Script for traffic sign detection on images.
- `trafficdetection_onnx.py`: ONNX model conversion and inference.
- `trafficdetection_ultra.py`: Main script for real-time traffic sign detection.
- `trafficdetection_ultra_extra.py`: Additional functionalities for detection.

## Usage
### Run Detection on Images
```bash
python trafficdetection_images_ultra.py --source path/to/image.jpg --weights best.pt
```

### Run Detection on Video
```bash
python trafficdetection_ultra.py --source path/to/video.mp4 --weights best.pt
```

### Run Detection Using Webcam
```bash
python trafficdetection_ultra.py --source 0 --weights best.pt
```

### Convert Model to ONNX Format
```bash
python trafficdetection_onnx.py --weights best.pt
```

## Model Details
- **YOLO version:** YOLOv8n
- **Framework:** PyTorch
- **Input size:** Standard YOLO input size
- **Dataset Source:** [Traffic Signs and Traffic Lights](https://universe.roboflow.com/traffic-signs-detection-aojvk/traffic-signs-and-traffic-lights)





