import cv2
from ultralytics import YOLO
import sys

def test_yolo_detection():
    # Path to one of your videos
    video_path = "cctv_samples/pass3.mp4"
    model_path = "yolo11n.pt"
    
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
        
    cv2.namedWindow("YOLO Detection Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        frame_count += 1
        
        # Run pure detection (predict, not track)
        # Using the same settings as your config (imgsz=1280) and low confidence (0.10)
        results = model.predict(
            frame, 
            classes=[0], # Person only
            conf=0.10,   # Low confidence threshold
            imgsz=1280,  # High resolution
            verbose=False
        )
        
        # Draw bounding boxes directly from YOLO's results
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw Box (Blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Draw Confidence score
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
        cv2.imshow("YOLO Detection Test", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolo_detection()
