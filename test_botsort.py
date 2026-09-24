import cv2
from ultralytics import YOLO
import sys

def test_botsort_tracking():
    # Path to one of your videos
    video_path = "cctv_samples/pass3.mp4"
    model_path = "yolo11m.pt"
    
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
        
    cv2.namedWindow("BoT-SORT Tracking Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        frame_count += 1
        
        # Run pure tracking (track, with botsort.yaml)
        results = model.track(
            frame, 
            classes=[0],          # Person only
            conf=0.10,            # Low confidence threshold
            imgsz=1280,           # High resolution
            tracker="botsort.yaml",
            persist=True,         # Extremely important for tracking!
            verbose=False
        )
        
        # Draw bounding boxes and Track IDs
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Check if IDs exist
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()
                for box, conf, tid in zip(boxes, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Trk: {int(tid)} Conf: {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # If tracker fails to assign IDs, draw yellow boxes (like our main app fallback)
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"NO TRACK ID Conf: {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
        cv2.imshow("BoT-SORT Tracking Test", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_botsort_tracking()
