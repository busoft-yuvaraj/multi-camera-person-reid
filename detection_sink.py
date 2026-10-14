from ultralytics import YOLO
import cv2

# Path to your sink detection model
MODEL_PATH = "models/yolo11n(best).pt"

# Image/video to test
SOURCE = "cctv_samples/handwash-sink-detection.png"   # Change this to your image/video path

# Load model
model = YOLO(MODEL_PATH)

# Run detection
results = model.predict(
    source=SOURCE,
    conf=0.10,
    
    save=True,
    show=True
)

# Print detections
for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        print("❌ No sink detected")
        continue

    detected = False

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

        if class_name.lower() == "sink":
            detected = True

    if detected:
        print("✅ SINK DETECTED")
    else:
        print("❌ SINK NOT DETECTED")