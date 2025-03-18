import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Load and preprocess the video
video_path = "/home/mona/Traffic detection/videos/vidd1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Error opening video stream or file.")

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the confidence threshold and NMS threshold
confidence_threshold = 0.3
nms_threshold = 0.4

# Class names (translated)
class_names = [
    'Att-STOP', 'Att-yield', 'Att-danger', 'Att-landslide', 
    'Att-priority intersection', 'Att-wildlife crossing', 
    'Att-children crossing', 'Att-pedestrian crossing', 'Att-speed bumps', 
    'Att-roundabout', 'Att-dual carriageway', 'Att-slippery road', 
    'Att-priority road', 'Att-successive curves', 'Att-construction', 
    'Att-right turn', 'Att-left turn', 'Indic-highway', 
    'Indic-one way traffic', 'Indic-parking', 'Indic-pedestrian crossing', 
    'Indic-bus station', 'Inter-no parking or stopping', 'Inter-no overtaking', 
    'Inter-no U-turn', 'Inter-turn right', 'Inter-turn left', 
    'Inter-directional traffic', 'Inter-parking prohibited', 'Inter-speed limit 100km/h', 
    'Inter-speed limit 120km/h', 'Inter-speed limit 20km/h', 'Inter-speed limit 30km/h', 
    'Inter-speed limit 40km/h', 'Inter-speed limit 50km/h', 'Inter-speed limit 60km/h', 
    'Inter-speed limit 80km/h', 'Oblig-turn right', 'Oblig-turn left', 
    'Oblig-straight ahead', 'Oblig-straight or turn right', 
    'Oblig-straight or turn left', 'Oblig-turn right at roundabout', 
    'Oblig-turn left at roundabout'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    # Run inference
    outputs = session.run(None, {"images": input_image})[0]

    boxes, confidences, class_ids = [], [], []

    # Post-process the outputs
    for i in range(outputs.shape[2]):  # Loop over all detections
        row = outputs[0, :, i]  # Extract one detection

        x, y, w, h = row[:4]  # First 4 values are bounding box
        confidence = max(row[5:])  # Highest class confidence
        class_id = np.argmax(row[5:])  # Class with max probability

        if confidence > confidence_threshold:
            # Convert YOLO format (center x, center y, w, h) to (x1, y1, x2, y2)
            x1 = int((x - w / 2) * width / 640)
            y1 = int((y - h / 2) * height / 640)
            x2 = int((x + w / 2) * width / 640)
            y2 = int((y + h / 2) * height / 640)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add the class label
            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with the bounding boxes and labels
    cv2.imshow("Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
