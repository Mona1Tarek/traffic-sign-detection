# Traffic Sign Detection

## Overview
This project implements a Traffic Sign Detection system using YOLO for object detection. The model is capable of detecting and classifying 45 different traffic signs in real-time from video streams or images.

## Features
  * Real-time traffic sign detection using YOLO.
  * Supports video input, webcam feeds, and image files.
  * Classifies traffic signs into 45 categories.
  * Outputs processed video with detected signs.

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

## Usage

### Run Detection on Images
```bash
python main.py --source path/to/image.jpg --weights best.pt
```

### Run Detection on Video
```bash
python main.py --source path/to/video.mp4 --weights best.pt
```

### Run Detection Using Webcam
```bash
python main.py --source 0 --weights best.pt
```

### Convert Model to ONNX Format
```bash
python optimize_onnx.py --weights best.pt
```

### Optimized Model
The optimized ONNX model is stored as `best_optimized.onnx`.

## Model Details
  * YOLO version: YOLOv8n
  * Framework: PyTorch
  * Input size: Standard YOLO input size
  * Dataset Source: [Traffic Signs and Traffic Lights](https://universe.roboflow.com)
