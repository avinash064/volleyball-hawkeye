"""
Simple Player Movement Tracker
Simplified version - just tracks and visualizes player movement
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque, defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RTDETRDetector:
    """RT-DETR object detector"""
    
    def __init__(self, weights_path: str, device='cuda'):
        logger.info(f"Loading RT-DETR: {weights_path}")
        from ultralytics import RTDETR
        self.model = RTDETR(weights_path)
        self.model.to(device)
        logger.info("RT-DETR loaded successfully")
    
    def detect(self, frame: np.ndarray, conf_threshold=0.3):
        """Run detection and return player bounding boxes"""
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        players = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            class_names = results.names
            
            for box, conf, cls in zip(boxes, confidences, classes):
                class_name = class_names[cls].lower()
                if 'player' in class_name:
                    x1, y1, x2, y2 = box
                    players.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
        
        return players

class SimpleTracker:
    """Simple IoU-based tracker"""
    
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.colors = {}  # Track ID -> color
    
    def update(self, detections):
        """Update tracks with new detections"""
        if not detections:
            # Age out old tracks
            for track in list(self.tracks.values()):
                track['age'] += 1
            self.tracks = {tid: t for tid, t in self.tracks.items() if t['age'] < self.max_age}
            return []
        
        matched_tracks = []
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = det['bbox']
                self.tracks[best_track_id]['age'] = 0
                det['track_id'] = best_track_id
                det['color'] = self.colors[best_track_id]
            else:
                # Create new track
                det['track_id'] = self.next_id
                # Assign random color
                color = tuple(np.random.randint(50, 255, 3).tolist())
                self.colors[self.next_id] = color
                det['color'] = color
                
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'age': 0
                }
                self.next_id += 1
            
            matched_tracks.append(det)
        
        # Age unmatched tracks
        active_ids = {det['track_id'] for det in matched_tracks}
        for track_id in self.tracks:
            if track_id not in active_ids:
                self.tracks[track_id]['age'] += 1
        
        return matched_tracks
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class MovementTracker:
    """Track player movement trajectories"""
    
    def __init__(self, max_trail_length=30):
        self.trajectories = defaultdict(lambda: deque(maxlen=max_trail_length))
    
    def update(self, tracked_players):
        """Update trajectories with new positions"""
        for player in tracked_players:
            track_id = player['track_id']
            bbox = player['bbox']
            # Use bottom-center (feet position)
            center = ((bbox[0] + bbox[2]) // 2, bbox[3])
            self.trajectories[track_id].append(center)
    
    def draw_trajectories(self, frame, tracked_players):
        """Draw movement trails on frame"""
        for player in tracked_players:
            track_id = player['track_id']
            color = player['color']
            
            if track_id in self.trajectories:
                trail = list(self.trajectories[track_id])
                
                # Draw trail
                for i in range(len(trail) - 1):
                    # Fade older positions
                    alpha = (i + 1) / len(trail)
                    thickness = int(2 + alpha * 3)
                    
                    cv2.line(frame, trail[i], trail[i + 1], color, thickness)
                
                # Draw current position circle
                if trail:
                    cv2.circle(frame, trail[-1], 5, color, -1)

def track_player_movement(video_path, weights_path, output_path, device='cuda'):
    """
    Simple player movement tracking
    
    Args:
        video_path: Input video
        weights_path: RT-DETR weights
        output_path: Output video
        device: 'cuda' or 'cpu'
    """
    logger.info("="*60)
    logger.info("SIMPLE PLAYER MOVEMENT TRACKER")
    logger.info("="*60)
    
    # Initialize
    detector = RTDETRDetector(weights_path, device=device)
    tracker = SimpleTracker(max_age=30)
    movement = MovementTracker(max_trail_length=50)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"\nVideo: {width}x{height} @ {fps} FPS")
    logger.info(f"Total frames: {total_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"\nProcessing...\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Detect players
            players = detector.detect(frame, conf_threshold=0.25)
            
            # Track players
            tracked = tracker.update(players)
            
            # Update movement trails
            movement.update(tracked)
            
            # Draw on frame
            output_frame = frame.copy()
            
            # Draw trajectories first (behind boxes)
            movement.draw_trajectories(output_frame, tracked)
            
            # Draw bounding boxes and IDs
            for player in tracked:
                bbox = player['bbox']
                track_id = player['track_id']
                color = player['color']
                
                # Draw bounding box
                cv2.rectangle(output_frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color, 3)
                
                # Draw ID label
                label = f"Player {track_id}"
                (text_w, text_h), _ = cv2.getTextSize(label, 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                                       0.7, 2)
                
                # Background for text
                cv2.rectangle(output_frame, 
                            (bbox[0], bbox[1] - text_h - 10), 
                            (bbox[0] + text_w + 10, bbox[1]), 
                            color, -1)
                
                # Text
                cv2.putText(output_frame, label, 
                           (bbox[0] + 5, bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 255, 255), 2)
            
            # Add info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Players: {len(tracked)}"
            cv2.putText(output_frame, info_text, 
                       (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0), 2)
            
            # Write output
            out.write(output_frame)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        cap.release()
        out.release()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Complete! Processed {frame_count}/{total_frames} frames")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    # Configuration
    WEIGHTS_PATH = r"C:\Users\xghostrider\Downloads\best(2).pt"
    VIDEO_PATH = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\input_videos\Video1.mp4"
    OUTPUT_PATH = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\player_movement.mp4"
    
    # Check paths
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"Weights not found: {WEIGHTS_PATH}")
        exit(1)
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f"Video not found: {VIDEO_PATH}")
        exit(1)
    
    # Run
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    track_player_movement(
        video_path=VIDEO_PATH,
        weights_path=WEIGHTS_PATH,
        output_path=OUTPUT_PATH,
        device=device
    )
