import cv2
import numpy as np
from ultralytics import YOLO

def detect_signs_from_camera():
    model = YOLO('best.pt')
    threshold = 0.5

    # Class name mapping (French to English)
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

    # Set up color per class
    np.random.seed(42)
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in model.names.values()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_signs = []

        for box in results.boxes:
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            if score > threshold:
                class_name = results.names[class_id]
                english_name = class_name_mapping.get(class_name, class_name)
                frame_signs.append(english_name)

                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(class_name, (0, 255, 0))
                label = f"{english_name} {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 8, y1), color, -1)
                cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Traffic Sign Detection", frame)

        # Yield detected signs for this frame
        yield frame_signs

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    for signs in detect_signs_from_camera():
        print("Detected signs:", signs)
