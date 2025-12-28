"""
Inference Script for Volleyball Tracking

Process volleyball videos with detection, tracking, team classification,
and court projection. Works with both fine-tuned and COCO-pretrained models.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.detector import VolleyballDetector
from tracking.tracker import ByteTrack
from tracking.team_classifier import TeamClassifier
from projection.homography import CourtHomography


def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball tracking inference")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8m.pt",
        help="YOLO model path (fine-tuned or COCO)"
    )
    parser.add_argument(
        "--conf-player",
        type=float,
        default=0.35,
        help="Player confidence threshold"
    )
    parser.add_argument(
        "--conf-ball",
        type=float,
        default=0.15,
        help="Ball confidence threshold (lower for recall)"
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable ByteTrack"
    )
    parser.add_argument(
        "--classify-teams",
        action="store_true",
        help="Enable team classification"
    )
    parser.add_argument(
        "--project-court",
        action="store_true",
        help="Enable court projection (requires keypoints)"
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default="configs/court_keypoints.json",
        help="Court keypoints for homography"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save output video"
    )
    return parser.parse_args()


def draw_detections(frame, detections, tracks=None, team_labels=None):
    """Draw bounding boxes and labels on frame"""
    overlay = frame.copy()
    
    # Team colors
    team_colors = {
        "Team A": (255, 100, 100),
        "Team B": (100, 100, 255),
        "Referee": (100, 255, 100),
        "Unknown": (150, 150, 150)
    }
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["conf"]
        class_name = det["class"]
        
        # Get color
        if team_labels is not None and i < len(team_labels):
            color = team_colors.get(team_labels[i], (0, 255, 0))
            label = f"{team_labels[i]} {conf:.2f}"
        else:
            color = (0, 255, 0) if class_name == "player" else (0, 0, 255)
            label = f"{class_name} {conf:.2f}"
        
        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(
            overlay, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Draw track ID if available
        if tracks is not None and i < len(tracks):
            track_id = tracks[i]["track_id"]
            cv2.putText(
                overlay, f"ID:{track_id}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    
    return overlay


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🏐 Volleyball Tracking System")
    print("=" * 60)
    
    # Initialize detector
    print("\n[1/5] Initializing detector...")
    detector = VolleyballDetector(
        model_path=args.model,
        conf_player=args.conf_player,
        conf_ball=args.conf_ball
    )
    
    # Initialize tracker
    tracker = None
    if args.track:
        print("[2/5] Initializing tracker...")
        tracker = ByteTrack(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
    
    # Initialize team classifier
    team_classifier = None
    if args.classify_teams:
        print("[3/5] Initializing team classifier...")
        team_classifier = TeamClassifier(n_clusters=3, color_space="HSV")
    
    # Initialize court homography
    court_homo = None
    if args.project_court:
        print("[4/5] Initializing court projection...")
        if Path(args.keypoints).exists():
            court_homo = CourtHomography(args.keypoints)
        else:
            print(f"⚠️ Keypoints not found: {args.keypoints}")
            print("   Court projection disabled")
    
    # Open video
    print("[5/5] Processing video...")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {args.video}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {args.video}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    
    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_writer = None
    if args.save_video:
        output_path = output_dir / f"{Path(args.video).stem}_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
    
    # Collect crops for team classification
    player_crops_for_fitting = []
    
    print("\n" + "=" * 60)
    print("Processing frames...")
    print("=" * 60)
    
    frame_idx = 0
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame, return_crops=args.classify_teams)
        
        # Track
        if tracker:
            tracks = tracker.update(detections)
        else:
            tracks = None
        
        # Collect crops for fitting team classifier
        if team_classifier and not team_classifier.is_fitted:
            for det in detections:
                if det["class"] == "player" and "crop" in det:
                    player_crops_for_fitting.append(det["crop"])
            
            # Fit after 50 frames
            if frame_idx == 50 and len(player_crops_for_fitting) > 0:
                print("\n\nFitting team classifier...")
                team_classifier.fit(player_crops_for_fitting)
        
        # Classify teams
        team_labels = None
        if team_classifier and team_classifier.is_fitted:
            player_dets = [d for d in detections if d["class"] == "player"]
            if len(player_dets) > 0 and all("crop" in d for d in player_dets):
                crops = [d["crop"] for d in player_dets]
                team_labels = team_classifier.predict_batch(crops)
        
        # Draw
        output_frame = draw_detections(frame, detections, tracks, team_labels)
        
        # Save frame
        if video_writer:
            video_writer.write(output_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if video_writer:
        video_writer.release()
    
    print("\n" + "=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)
    
    if args.save_video:
        print(f"\nOutput saved to: {output_path}")
    
    print(f"\nProcessed {frame_idx} frames")
    print(f"Average FPS: {frame_idx / (total_frames / fps):.2f}")


if __name__ == "__main__":
    main()
