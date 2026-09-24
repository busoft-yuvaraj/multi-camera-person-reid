import os
import cv2
import argparse

def extract_frames(input_dir, output_dir, frame_interval):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No videos found in '{input_dir}'.")
        return

    print(f"Found {len(video_files)} videos. Extracting 1 frame every {frame_interval} frames...")
    
    total_extracted = 0
    
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue
            
        frame_count = 0
        extracted_from_video = 0
        video_name = os.path.splitext(video_file)[0]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Format: original_video_name_frame_00123.jpg
                out_name = f"{video_name}_frame_{frame_count:05d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                
                cv2.imwrite(out_path, frame)
                extracted_from_video += 1
                total_extracted += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {extracted_from_video} frames from {video_file}")
        
    print(f"\nDone! Extracted a total of {total_extracted} frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from CCTV videos for annotation")
    parser.add_argument("--input_dir", type=str, default="cctv_samples", help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="cctv_frames", help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=30, help="Extract 1 frame every N frames (30 = approx 2 frames per second, good for fast movement)")
    
    args = parser.parse_args()
    extract_frames(args.input_dir, args.output_dir, args.interval)
