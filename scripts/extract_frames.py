"""
Frame Extraction Tool for Volleyball Videos

Extract frames from volleyball videos for annotation and training.
Supports intelligent sampling to get diverse training data.
"""

import cv2
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json


def extract_frames(
    video_path: str,
    output_dir: str,
    sampling_rate: int = 15,
    max_frames: int = 500,
    min_motion_threshold: float = 10.0,
    save_metadata: bool = True
):
    """
    Extract frames from volleyball video with intelligent sampling.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        sampling_rate: Extract every Nth frame (e.g., 15 = ~0.5s @ 30fps)
        max_frames: Maximum number of frames to extract
        min_motion_threshold: Minimum motion to consider frame (skips static frames)
        save_metadata: Save frame metadata (timestamps, indices)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("=" * 60)
    print("🏐 Volleyball Frame Extraction")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"Sampling rate: Every {sampling_rate} frames (~{sampling_rate/fps:.2f}s)")
    print(f"Max frames: {max_frames}")
    print("=" * 60)
    
    frame_idx = 0
    saved_count = 0
    prev_frame = None
    metadata = []
    
    pbar = tqdm(total=min(total_frames, max_frames * sampling_rate), desc="Extracting")
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on sampling rate
        if frame_idx % sampling_rate == 0:
            # Calculate motion (skip very static frames)
            skip_frame = False
            if prev_frame is not None:
                motion = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                if motion < min_motion_threshold:
                    skip_frame = True
            
            if not skip_frame:
                # Save frame
                frame_name = f"frame_{saved_count:05d}.jpg"
                frame_path = output_path / frame_name
                cv2.imwrite(str(frame_path), frame)
                
                # Save metadata
                timestamp = frame_idx / fps if fps > 0 else 0
                metadata.append({
                    "filename": frame_name,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "resolution": [width, height]
                })
                
                saved_count += 1
                prev_frame = frame.copy()
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save metadata
    if save_metadata and metadata:
        metadata_path = output_path / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "video_path": str(video_path),
                "total_frames_extracted": saved_count,
                "sampling_rate": sampling_rate,
                "fps": fps,
                "frames": metadata
            }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Extraction Complete!")
    print("=" * 60)
    print(f"Extracted: {saved_count} frames")
    print(f"Saved to: {output_path}")
    if save_metadata:
        print(f"Metadata: {metadata_path}")
    print("\nNext steps:")
    print("  1. Review extracted frames")
    print("  2. Annotate using CVAT, LabelImg, or Roboflow")
    print("  3. Export annotations in YOLO format")
    print("  4. Train model with: python scripts/train.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from volleyball video")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/volleyball/images/train",
        help="Output directory for frames"
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=15,
        help="Extract every Nth frame (default: 15)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to extract (default: 500)"
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=10.0,
        help="Minimum motion threshold to skip static frames (default: 10.0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        sampling_rate=args.sampling_rate,
        max_frames=args.max_frames,
        min_motion_threshold=args.min_motion
    )


if __name__ == "__main__":
    main()
