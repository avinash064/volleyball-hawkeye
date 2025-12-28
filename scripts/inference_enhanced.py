"""
Enhanced Inference Script with Re-ID and Ball Trajectory

Processes volleyball videos with:
- Detection (YOLO)
- Re-ID enhanced tracking (ByteTrack + appearance features)
- Team classification (jersey colors)  
- Ball trajectory prediction
- Court projection (optional)
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
from tracking.reid import ReIDFeatureExtractor, ReIDMatcher
from tracking.trajectory import BallTrajectoryPredictor, draw_ball_trajectory
from projection.homography import CourtHomography
from visualization.render import draw_bbox, draw_trajectory, create_side_by_side, add_info_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball tracking with Re-ID and trajectory")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--model", type=str, default="models/yolov8m.pt", help="YOLO model path")
    
    # Detection
    parser.add_argument("--conf-player", type=float, default=0.35, help="Player confidence")
    parser.add_argument("--conf-ball", type=float, default=0.15, help="Ball confidence")
    
    # Tracking
    parser.add_argument("--track", action="store_true", default=True, help="Enable tracking")
    parser.add_argument("--use-reid", action="store_true", help="Enable Re-ID tracking")
    parser.add_argument("--reid-model", type=str, default="resnet18", 
                       choices=["resnet18", "mobilenet_v3_small"], help="Re-ID model")
    
    # Team classification
    parser.add_argument("--classify-teams", action="store_true", help="Enable team classification")
    
    # Ball trajectory
    parser.add_argument("--predict-trajectory", action="store_true", help="Enable ball trajectory prediction")
    parser.add_argument("--trajectory-length", type=int, default=30, help="Trajectory prediction frames")
    
    # Court projection
    parser.add_argument("--project-court", action="store_true", help="Enable court projection")
    parser.add_argument("--keypoints", type=str, default="configs/court_keypoints.json", 
                       help="Court keypoints")
    
    # Visualization
    parser.add_argument("--show-trajectories", action="store_true", help="Show player trajectories")
    parser.add_argument("--trajectory-history", type=int, default=30, help="Trajectory trail length")
    
    # Output
    parser.add_argument("--save-video", action="store_true", default=True, help="Save output video")
    parser.add_argument("--display", action="store_true", help="Display video while processing")
    
    return parser.parse_args()


def draw_enhanced_frame(
    frame,
    detections,
    tracks=None,
    team_labels=None,
    ball_trajectory=None,
    ball_prediction=None,
    show_trajectories=False,
    info=None
):
    """Draw all annotations on frame"""
    output = frame.copy()
    
    # Team colors
    team_colors = {
        "Team A": (255, 100, 100),
        "Team B": (100, 100, 255),
        "Referee": (100, 255, 100),
        "Unknown": (150, 150, 150)
    }
    
    # Draw bounding boxes and tracks
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["conf"]
        class_name = det["class"]
        
        # Skip ball for now (will draw trajectory separately)
        if class_name == "ball":
            continue
        
        # Get color and label
        if team_labels is not None and i < len(team_labels):
            color = team_colors.get(team_labels[i], (0, 255, 0))
            label = f"{team_labels[i]} {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = f"{class_name} {conf:.2f}"
        
        # Draw bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(output, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw track ID and trajectory
        if tracks is not None:
            # Find matching track
            for track in tracks:
                track_bbox = track.get("bbox", [])
                if len(track_bbox) == 4:
                    tx1, ty1, tx2, ty2 = track_bbox
                    # Simple overlap check
                    if abs(x1 - tx1) < 20 and abs(y1 - ty1) < 20:
                        track_id = track["track_id"]
                        cv2.putText(output, f"ID:{track_id}", (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw trajectory
                        if show_trajectories and "trajectory" in track:
                            traj = track["trajectory"]
                            if len(traj) > 1:
                                points = np.array(traj, dtype=np.int32)
                                for j in range(len(points) - 1):
                                    alpha = (j + 1) / len(points)
                                    fade_color = tuple(int(c * alpha) for c in color)
                                    cv2.line(output, tuple(points[j]), tuple(points[j + 1]),
                                            fade_color, 2)
                        break
    
    # Draw ball trajectory
    if ball_trajectory is not None or ball_prediction is not None:
        output = draw_ball_trajectory(
            output,
            ball_trajectory if ball_trajectory else [],
            ball_prediction,
            history_color=(0, 255, 255),
            prediction_color=(0, 165, 255),
            thickness=2
        )
    
    # Draw info panel
    if info:
        output = add_info_panel(output, info, position="top-left")
    
    return output


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🏐 Enhanced Volleyball Tracking System")
    print("=" * 60)
    
    # Initialize detector
    print("\n[1/6] Initializing detector...")
    detector = VolleyballDetector(
        model_path=args.model,
        conf_player=args.conf_player,
        conf_ball=args.conf_ball
    )
    
    # Initialize tracker
    tracker = None
    reid_extractor = None
    reid_matcher = None
    
    if args.track:
        print("[2/6] Initializing tracker...")
        tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        
        if args.use_reid:
            print("  [+] Initializing Re-ID feature extractor...")
            reid_extractor = ReIDFeatureExtractor(
                model_name=args.reid_model,
                device="cuda:0"
            )
            reid_matcher = ReIDMatcher(
                reid_extractor,
                iou_weight=0.4,
                reid_weight=0.6
            )
            print("  [+] Re-ID tracking enabled!")
    
    # Initialize team classifier
    team_classifier = None
    if args.classify_teams:
        print("[3/6] Initializing team classifier...")
        team_classifier = TeamClassifier(n_clusters=3, color_space="HSV")
    
    # Initialize ball trajectory predictor
    ball_predictor = None
    if args.predict_trajectory:
        print("[4/6] Initializing trajectory predictor...")
        ball_predictor = BallTrajectoryPredictor(fps=30, use_kalman=True)
    
    # Initialize court homography
    court_homo = None
    if args.project_court and Path(args.keypoints).exists():
        print("[5/6] Initializing court projection...")
        court_homo = CourtHomography(args.keypoints)
    
    # Open video
    print("[6/6] Loading video...")
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
        output_path = output_dir / f"{Path(args.video).stem}_enhanced.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
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
        
        # Detect (with crops for Re-ID)
        need_crops = (reid_extractor is not None) or (team_classifier is not None)
        detections = detector.detect(frame, return_crops=need_crops)
        
        # Extract Re-ID features if enabled
        if reid_extractor and detections:
            player_dets = [d for d in detections if d["class"] == "player"]
            if player_dets:
                crops = [d.get("crop") for d in player_dets]
                features = reid_extractor.extract_features(crops)
                # Add features to detections
                for i, det in enumerate(player_dets):
                    if i < len(features):
                        det["features"] = features[i]
        
        # Track
        tracks = None
        if tracker:
            tracks = tracker.update(detections)
        
        # Update ball trajectory
        ball_trajectory = None
        ball_prediction = None
        if ball_predictor:
            ball_dets = [d for d in detections if d["class"] == "ball"]
            if ball_dets:
                # Use most confident ball detection
                ball_det = max(ball_dets, key=lambda x: x["conf"])
                ball_bbox = ball_det["bbox"]
                ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2,
                              (ball_bbox[1] + ball_bbox[3]) / 2)
                ball_predictor.update(ball_center)
                
                ball_trajectory = ball_predictor.get_trajectory()
                ball_prediction = ball_predictor.predict(args.trajectory_length)
        
        # Collect crops for team classifier fitting
        if team_classifier and not team_classifier.is_fitted:
            for det in detections:
                if det["class"] == "player" and "crop" in det:
                    player_crops_for_fitting.append(det["crop"])
            
            if frame_idx == 50 and len(player_crops_for_fitting) > 0:
                print("\nFitting team classifier...")
                team_classifier.fit(player_crops_for_fitting)
        
        # Classify teams
        team_labels = None
        if team_classifier and team_classifier.is_fitted:
            player_dets = [d for d in detections if d["class"] == "player"]
            if player_dets and all("crop" in d for d in player_dets):
                crops = [d["crop"] for d in player_dets]
                team_labels = team_classifier.predict_batch(crops)
        
        # Create info panel
        info = {
            "Frame": f"{frame_idx}/{total_frames}",
            "Players": sum(1 for d in detections if d["class"] == "player"),
            "Ball": "✓" if any(d["class"] == "ball" for d in detections) else "✗",
        }
        if args.use_reid:
            info["Re-ID"] = "Active"
        if ball_predictor and ball_prediction:
            info["Trajectory"] = f"{len(ball_prediction)} frames"
        
        # Draw enhanced frame
        output_frame = draw_enhanced_frame(
            frame,
            detections,
            tracks,
            team_labels,
            ball_trajectory,
            ball_prediction,
            show_trajectories=args.show_trajectories,
            info=info
        )
        
        # Save/display
        if video_writer:
            video_writer.write(output_frame)
        
        if args.display:
            cv2.imshow("Volleyball Tracking", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if video_writer:
        video_writer.release()
    if args.display:
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)
    
    if args.save_video:
        print(f"\nOutput: {output_path}")
    print(f"Processed: {frame_idx} frames")


if __name__ == "__main__":
    main()
