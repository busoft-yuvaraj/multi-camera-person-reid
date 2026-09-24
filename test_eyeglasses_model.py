import os
import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test custom YOLO model on images")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--input_dir", type=str, default="test_images", help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold for detection")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for inference")
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get all image files in input directory
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.webp'))]
    
    if not image_files:
        print(f"\nNo images found in '{args.input_dir}/'.")
        print(f"Please add some sample images to the '{args.input_dir}' folder and run this script again.")
        return

    print(f"\nFound {len(image_files)} images. Running inference...\n")

    for img_name in image_files:
        img_path = os.path.join(args.input_dir, img_name)
        
        # Run inference with specified confidence and image size
        results = model.predict(
            source=img_path,
            conf=args.conf,
            imgsz=args.imgsz
        )
        
        for r in results:
            # Render the results on the image
            im_array = r.plot()  # plots BGR numpy array of predictions
            
            # Save the result
            save_path = os.path.join(args.output_dir, f"result_{img_name}")
            cv2.imwrite(save_path, im_array)
            
            print(f"Processed: {img_name}")
            print(f"  - Detected {len(r.boxes)} objects.")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"  - Found: {cls_name} (Confidence: {conf:.2f})")
            print(f"  - Saved result to: {save_path}\n")

if __name__ == "__main__":
    main()
