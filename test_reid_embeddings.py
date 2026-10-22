"""
Interactive ReID Embedding Comparison and Visualization Tool
============================================================
This script demonstrates how OSNet ReID embeddings are extracted, how cosine
similarity is computed, and why two people with different shirts can still have
elevated similarity scores under CCTV conditions.

Usage:
  # 1. Run automatic extraction from CCTV video:
  python test_reid_embeddings.py --video D:/cctv_samples/test8/pantry.mp4

  # 2. Compare custom image files:
  python test_reid_embeddings.py --images crop1.jpg crop2.jpg crop3.jpg

  # 3. Quick test mode (auto-detects video or generates demo crops):
  python test_reid_embeddings.py
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure local app modules can be imported
sys.path.insert(0, os.path.abspath("."))
from app.reid.extractor import OSNetExtractor


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D L2-normalized embedding vectors."""
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def extract_person_crops_from_video(video_path: str, max_crops: int = 5):
    """Extracts high-confidence person bounding box crops from a video using YOLO."""
    print(f"\n[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file at {video_path}")
        return []

    try:
        from ultralytics import YOLO
        detector = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"[WARNING] Could not load YOLO detector ({e}). Sampling full frames instead.")
        detector = None

    crops = []
    labels = []
    frame_indices = [30, 90, 180, 300, 450, 600, 800]

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if detector:
            results = detector(frame, classes=[0], verbose=False)
            boxes = results[0].boxes
            if boxes and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = xyxy
                    # Avoid tiny boxes
                    if (y2 - y1) < 80 or (x2 - x1) < 40:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop)
                    labels.append(f"F{f_idx}_Person{i+1}")
                    if len(crops) >= max_crops:
                        break
        else:
            # Fallback: take center crop
            h, w = frame.shape[:2]
            crop = frame[h//4:3*h//4, w//4:3*w//4]
            crops.append(crop)
            labels.append(f"Frame_{f_idx}")

        if len(crops) >= max_crops:
            break

    cap.release()
    return list(zip(crops, labels))


def generate_synthetic_crops():
    """Generates synthetic person silhouettes with different shirt colors for demonstration."""
    print("[INFO] Generating synthetic test crops (Blue Shirt, Pink/Purple Shirt, Gray Suit)...")
    crops_and_labels = []

    configs = [
        ("Person A (Blue Shirt)", (180, 80, 20), (40, 40, 40)),
        ("Person A (Blue Shirt - Low Light)", (140, 60, 15), (30, 30, 30)),
        ("Person B (Pink/Purple Shirt)", (140, 50, 180), (40, 40, 40)),
        ("Person C (Black Shirt)", (20, 20, 20), (40, 40, 40)),
    ]

    for label, shirt_bgr, pants_bgr in configs:
        img = np.ones((256, 128, 3), dtype=np.uint8) * 120  # Neutral background
        # Head
        cv2.circle(img, (64, 40), 22, (180, 200, 220), -1)
        # Shirt / Upper body
        cv2.rectangle(img, (34, 65), (94, 150), shirt_bgr, -1)
        # Pants / Lower body
        cv2.rectangle(img, (40, 150), (60, 240), pants_bgr, -1)
        cv2.rectangle(img, (68, 150), (88, 240), pants_bgr, -1)
        crops_and_labels.append((img, label))

    return crops_and_labels


def visualize_and_explain(crops, labels, embeddings, sim_matrix, output_path="reid_comparison_result.png"):
    """Creates a visual comparison image with crops and a similarity matrix heatmap."""
    n = len(crops)
    fig = plt.figure(figsize=(14, 8), dpi=100)
    plt.suptitle("Multi-Camera OSNet ReID Embedding Comparison", fontsize=16, fontweight="bold", y=0.98)

    # 1. Plot individual crops
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        rgb_crop = cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_crop)
        ax.set_title(labels[i], fontsize=10, fontweight="semibold")
        ax.axis("off")

    # 2. Plot Cosine Similarity Matrix Heatmap
    ax_mat = plt.subplot(2, 1, 2)
    cax = ax_mat.imshow(sim_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(cax, ax=ax_mat, fraction=0.03, pad=0.04, label="Cosine Similarity")

    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_mat.set_yticklabels(labels, fontsize=9)

    # Annotate matrix cells
    for i in range(n):
        for j in range(n):
            score = sim_matrix[i, j]
            color = "black" if 0.55 <= score <= 0.85 else "white"
            status = "MATCH" if score >= 0.70 else ("AMBIGUOUS" if score >= 0.55 else "DISTINCT")
            ax_mat.text(
                j, i, f"{score:.3f}\n({status})",
                ha="center", va="center", color=color, fontsize=8, fontweight="bold"
            )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the output figure
    plt.savefig(output_path)
    plt.close()
    print(f"\n[SUCCESS] Visual comparison saved to: {os.path.abspath(output_path)}")


def print_cli_summary(labels, sim_matrix):
    """Prints a formatted ASCII matrix table with threshold explanations."""
    n = len(labels)
    print("\n" + "=" * 75)
    print("PAIRWISE REID COSINE SIMILARITY MATRIX")
    print("=" * 75)
    header = f"{'':<25}" + "".join([f"{f'[{i}]':>10}" for i in range(n)])
    print(header)
    print("-" * 75)

    for i in range(n):
        row = f"[{i}] {labels[i][:20]:<21}"
        for j in range(n):
            score = sim_matrix[i, j]
            row += f"{score:>10.3f}"
        print(row)

    print("=" * 75)
    print("HOW TO INTERPRET THESE SCORES:")
    print("  • Score >= 0.70  [HIGH MATCH]   : High visual confidence of being the same person.")
    print("  • 0.55 <= Score < 0.70 [AMBIGUOUS] : Needs Grace Period & Camera Topology checks!")
    print("                                      (CCTV background / dark pants inflate this score)")
    print("  • Score < 0.55   [DISTINCT]     : Visually distinct identity.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Test and visualize OSNet ReID embeddings.")
    parser.add_argument("--video", type=str, default="", help="Path to video file to sample crops from")
    parser.add_argument("--images", nargs="+", help="Paths to custom images to compare")
    parser.add_argument("--output", type=str, default="reid_comparison_result.png", help="Output visualization image")
    args = parser.parse_args()

    # Step 1: Collect crops
    crops_and_labels = []
    if args.images:
        print(f"[INFO] Loading {len(args.images)} custom images...")
        for img_path in args.images:
            img = cv2.imread(img_path)
            if img is not None:
                crops_and_labels.append((img, os.path.basename(img_path)))
            else:
                print(f"[WARNING] Could not read image: {img_path}")
    elif args.video and os.path.exists(args.video):
        crops_and_labels = extract_person_crops_from_video(args.video)
    else:
        # Default fallback: check default video or generate synthetic
        default_video = "D:/cctv_samples/test8/pantry.mp4"
        if os.path.exists(default_video):
            crops_and_labels = extract_person_crops_from_video(default_video, max_crops=4)
        else:
            crops_and_labels = generate_synthetic_crops()

    if not crops_and_labels:
        print("[INFO] Fallback to synthetic crops...")
        crops_and_labels = generate_synthetic_crops()

    crops = [c[0] for c in crops_and_labels]
    labels = [c[1] for c in crops_and_labels]

    # Step 2: Initialize OSNet Extractor
    print(f"\n[INFO] Initializing OSNet Extractor (osnet_x1_0)...")
    extractor = OSNetExtractor(model_name="osnet_x1_0", device="cpu")

    # Step 3: Extract 512-dimensional embeddings
    print(f"[INFO] Extracting embeddings for {len(crops)} crops...")
    embeddings = extractor.extract(crops)

    # Step 4: Compute Pairwise Cosine Similarity Matrix
    n = len(embeddings)
    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = compute_cosine_similarity(embeddings[i], embeddings[j])

    # Step 5: Print Summary & Generate Visual Output
    print_cli_summary(labels, sim_matrix)
    visualize_and_explain(crops, labels, embeddings, sim_matrix, output_path=args.output)


if __name__ == "__main__":
    main()
