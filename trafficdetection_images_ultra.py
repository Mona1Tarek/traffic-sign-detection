import os
from ultralytics import YOLO
import cv2

# Define the directory containing images
IMAGES_DIR = os.path.join('.', 'images')  # Adjust the directory if necessary
image_path = os.path.join(IMAGES_DIR, '/home/mona/Traffic detection/images/44.jpg')  # Replace with your image filename

# Load the image
image = cv2.imread(image_path)

# Load your trained model
model_path = os.path.join('.', 'best.pt')  # Path to the trained model
model = YOLO(model_path)  # Load the custom trained YOLO model

# The threshold for detection confidence
threshold = 0.5

# Class name mapping (translated to English)
class_name_mapping = {
    'Att-STOP': 'STOP',
    'Att-cedez le passage': 'Give Way',
    'Att-danger': 'Danger',
    'Att-eboulement': 'Landslide',
    'Att-intersection ou vous etes prioritaire': 'Intersection, You Have Priority',
    'Att-passage animaux sauvages': 'Wildlife Crossing',
    'Att-passage enfants': 'Children Crossing',
    'Att-passage pietons': 'Pedestrian Crossing',
    'Att-ralentisseurs': 'Speed Bumps',
    'Att-rond point': 'Roundabout',
    'Att-route a double voie': 'Dual Carriageway',
    'Att-route glissante': 'Slippery Road',
    'Att-route prioritaire': 'Priority Road',
    'Att-succession de virages': 'Curves Ahead',
    'Att-travaux': 'Roadworks',
    'Att-virage a droite': 'Right Turn',
    'Att-virage a gauche': 'Left Turn',
    'Indic-autoroute': 'Motorway',
    'Indic-circulation a sens unique': 'One-way Traffic',
    'Indic-parking': 'Parking',
    'Indic-passage pietons': 'Pedestrian Crossing',
    'Indic-station bus': 'Bus Stop',
    'Inter-arret et stationnement': 'No Stopping or Parking',
    'Inter-de depasser': 'No Overtaking',
    'Inter-de faire demi-tour': 'No U-turn',
    'Inter-de tourner a droite': 'No Right Turn',
    'Inter-de tourner a gauche': 'No Left Turn',
    'Inter-sens': 'No Entry',
    'Inter-stationnement': 'No Parking',
    'Inter-vitesse limitee a -100km-h-': 'Speed Limit 100 km/h',
    'Inter-vitesse limitee a -120km-h-': 'Speed Limit 120 km/h',
    'Inter-vitesse limitee a -20km-h-': 'Speed Limit 20 km/h',
    'Inter-vitesse limitee a -30km-h-': 'Speed Limit 30 km/h',
    'Inter-vitesse limitee a -40km-h-': 'Speed Limit 40 km/h',
    'Inter-vitesse limitee a -50km-h-': 'Speed Limit 50 km/h',
    'Inter-vitesse limitee a -60km-h-': 'Speed Limit 60 km/h',
    'Inter-vitesse limitee a -80km-h-': 'Speed Limit 80 km/h',
    'Oblig-continuez a droite': 'Keep Right',
    'Oblig-continuez a gauche': 'Keep Left',
    'Oblig-continuez tout droit': 'Go Straight',
    'Oblig-continuez tout droit ou tournez a droite': 'Go Straight or Turn Right',
    'Oblig-continuez tout droit ou tournez a gauche': 'Go Straight or Turn Left',
    'Oblig-tournez a droite': 'Turn Right',
    'Oblig-tournez a gauche': 'Turn Left',
    'Oblig-tournez le rond point': 'Turn at Roundabout'
}

# Perform inference on the image
results = model(image)[0]

# Iterate over the detected boxes
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    # Only process detections above the threshold
    if score > threshold:
        # Get the original class name
        original_class_name = results.names[int(class_id)]

        # Get the English class name from the mapping
        english_class_name = class_name_mapping.get(original_class_name, original_class_name)

        # Draw the bounding box and label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, english_class_name, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the image with detections in a window
cv2.imshow('Detected Traffic Signs', image)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
