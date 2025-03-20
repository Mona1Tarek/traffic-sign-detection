import os
import cv2
import numpy as np
from ultralytics import YOLO

# Define directories
VIDEOS_DIR = os.path.join('.', 'videos')
input_video_path = os.path.join(VIDEOS_DIR, 'vidd7.mp4')
output_video_path = os.path.join(VIDEOS_DIR, 'output.mp4')

# Open the input video
cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Load your trained YOLO model
model_path = os.path.join('.', 'best.pt')
model = YOLO(model_path)

# Detection threshold
threshold = 0.5

# Class name mapping (French → English)
class_name_mapping = {
    'Att-STOP': 'STOP', 'Att-cedez le passage': 'Give Way', 'Att-danger': 'Danger',
    'Att-eboulement': 'Landslide', 'Att-intersection ou vous etes prioritaire': 'Intersection, You Have Priority',
    'Att-passage animaux sauvages': 'Wildlife Crossing', 'Att-passage enfants': 'Children Crossing',
    'Att-passage pietons': 'Pedestrian Crossing', 'Att-ralentisseurs': 'Speed Bumps',
    'Att-rond point': 'Roundabout', 'Att-route a double voie': 'Dual Carriageway',
    'Att-route glissante': 'Slippery Road', 'Att-route prioritaire': 'Priority Road',
    'Att-succession de virages': 'Curves Ahead', 'Att-travaux': 'Roadworks',
    'Att-virage a droite': 'Right Turn', 'Att-virage a gauche': 'Left Turn',
    'Indic-autoroute': 'Motorway', 'Indic-circulation a sens unique': 'One-way Traffic',
    'Indic-parking': 'Parking', 'Indic-passage pietons': 'Pedestrian Crossing',
    'Indic-station bus': 'Bus Stop', 'Inter-arret et stationnement': 'No Stopping or Parking',
    'Inter-de depasser': 'No Overtaking', 'Inter-de faire demi-tour': 'No U-turn',
    'Inter-de tourner a droite': 'No Right Turn', 'Inter-de tourner a gauche': 'No Left Turn',
    'Inter-sens': 'No Entry', 'Inter-stationnement': 'No Parking',
    'Inter-vitesse limitee a -100km-h-': 'Speed Limit 100 km/h',
    'Inter-vitesse limitee a -120km-h-': 'Speed Limit 120 km/h',
    'Inter-vitesse limitee a -20km-h-': 'Speed Limit 20 km/h',
    'Inter-vitesse limitee a -30km-h-': 'Speed Limit 30 km/h',
    'Inter-vitesse limitee a -40km-h-': 'Speed Limit 40 km/h',
    'Inter-vitesse limitee a -50km-h-': 'Speed Limit 50 km/h',
    'Inter-vitesse limitee a -60km-h-': 'Speed Limit 60 km/h',
    'Inter-vitesse limitee a -80km-h-': 'Speed Limit 80 km/h',
    'Oblig-continuez a droite': 'Keep Right', 'Oblig-continuez a gauche': 'Keep Left',
    'Oblig-continuez tout droit': 'Go Straight', 'Oblig-continuez tout droit ou tournez a droite': 'Go Straight or Turn Right',
    'Oblig-continuez tout droit ou tournez a gauche': 'Go Straight or Turn Left',
    'Oblig-tournez a droite': 'Turn Right', 'Oblig-tournez a gauche': 'Turn Left',
    'Oblig-tournez le rond point': 'Turn at Roundabout'
}

# Assign unique colors for each class
np.random.seed(42)  # Ensure consistent colors
colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in model.names.values()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # Exit if no more frames
        print("End of video reached.")
        break

    results = model(frame)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            original_class_name = results.names[int(class_id)]
            english_class_name = class_name_mapping.get(original_class_name, original_class_name)
            color = colors.get(original_class_name, (0, 255, 0))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{english_class_name} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - text_h - 8), (int(x1) + text_w + 8, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 4, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)  # Ensure every frame is written
    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_video_path}")
