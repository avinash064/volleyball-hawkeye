"""
Volleyball Hawk-Eye Tactical Intelligence System
Complete integrated pipeline for volleyball tactical analysis
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 0: GEOMETRY & COURT MODEL
# ============================================================================

class VolleyballCourtConfiguration:
    """Volleyball court geometry and configuration"""
    
    # Court dimensions in centimeters (official FIVB standards)
    COURT_LENGTH = 1800  # 18 meters
    COURT_WIDTH = 900    # 9 meters
    NET_X = 900          # Net at center (9m from either end)
    
    # Attack lines (3m from net)
    ATTACK_LINE_DISTANCE = 300
    
    # Court vertices (for homography)
    VERTICES = np.array([
        [0, 0],                    # Top-left
        [COURT_LENGTH, 0],         # Top-right
        [COURT_LENGTH, COURT_WIDTH],  # Bottom-right
        [0, COURT_WIDTH]           # Bottom-left
    ], dtype=np.float32)
    
    # Net line
    NET_LINE = [(NET_X, 0), (NET_X, COURT_WIDTH)]
    
    # Attack lines
    LEFT_ATTACK_LINE = [(NET_X - ATTACK_LINE_DISTANCE, 0), 
                        (NET_X - ATTACK_LINE_DISTANCE, COURT_WIDTH)]
    RIGHT_ATTACK_LINE = [(NET_X + ATTACK_LINE_DISTANCE, 0), 
                         (NET_X + ATTACK_LINE_DISTANCE, COURT_WIDTH)]
    
    # Colors for visualization
    COURT_COLOR = (139, 69, 19)  # Brown
    LINE_COLOR = (255, 255, 255)  # White
    NET_COLOR = (200, 200, 200)   # Light gray
    
    TEAM_A_COLOR = (255, 0, 0)    # Blue
    TEAM_B_COLOR = (0, 0, 255)    # Red
    BALL_COLOR = (0, 255, 255)    # Yellow
    REFEREE_COLOR = (128, 128, 128)  # Gray

def draw_volleyball_pitch(width=900, height=1800):
    """
    Draw a top-down 2D volleyball court
    
    Args:
        width: Canvas width in pixels
        height: Canvas height in pixels
    
    Returns:
        court_img: Rendered court as numpy array (BGR)
    """
    # Create canvas
    court_img = np.full((int(height), int(width), 3), 
                        VolleyballCourtConfiguration.COURT_COLOR, 
                        dtype=np.uint8)
    
    config = VolleyballCourtConfiguration
    
    # Draw court outline
    cv2.rectangle(court_img, (0, 0), (width-1, height-1), 
                  config.LINE_COLOR, thickness=3)
    
    # Draw centerline (net)
    net_y = height // 2
    cv2.line(court_img, (0, net_y), (width, net_y), 
             config.NET_COLOR, thickness=5)
    
    # Draw attack lines
    attack_offset = int(height * 0.167)  # 3m / 18m = 0.167
    cv2.line(court_img, (0, net_y - attack_offset), 
             (width, net_y - attack_offset), 
             config.LINE_COLOR, thickness=2)
    cv2.line(court_img, (0, net_y + attack_offset), 
             (width, net_y + attack_offset), 
             config.LINE_COLOR, thickness=2)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(court_img, "TEAM A", (int(width*0.35), int(height*0.25)), 
                font, 1.5, (255, 255, 255), 2)
    cv2.putText(court_img, "TEAM B", (int(width*0.35), int(height*0.75)), 
                font, 1.5, (255, 255, 255), 2)
    cv2.putText(court_img, "NET", (int(width*0.42), net_y - 10), 
                font, 1, (255, 255, 255), 2)
    
    return court_img

# ============================================================================
# PHASE 1: DETECTION & TRACKING
# ============================================================================

class RTDETRDetector:
    """RT-DETR object detector wrapper"""
    
    def __init__(self, weights_path: str, device='cuda'):
        """
        Initialize RT-DETR detector
        
        Args:
            weights_path: Path to trained weights
            device: 'cuda' or 'cpu'
        """
        self.device = device
        logger.info(f"Loading RT-DETR from {weights_path}")
        
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(weights_path)
            self.model.to(device)
            logger.info("RT-DETR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RT-DETR: {e}")
            raise
    
    def detect(self, frame: np.ndarray, conf_threshold=0.3):
        """
        Run detection on frame
        
        Args:
            frame: Input image (BGR)
            conf_threshold: Confidence threshold
        
        Returns:
            detections: Dict with 'players', 'referees', 'balls'
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        detections = {
            'players': [],
            'referees': [],
            'balls': []
        }
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            class_names = results.names
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                class_name = class_names[cls].lower()
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name
                }
                
                if 'player' in class_name:
                    detections['players'].append(detection)
                elif 'referee' in class_name:
                    detections['referees'].append(detection)
                elif 'volleyball' in class_name or 'ball' in class_name:
                    detections['balls'].append(detection)
        
        return detections

class SimpleTracker:
    """
    Simple tracking based on IoU matching
    (Fallback if ByteTrack not available)
    """
    
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
    
    def update(self, detections: List[dict]):
        """Update tracks with new detections"""
        if not detections:
            # Age out old tracks
            self.tracks = {tid: track for tid, track in self.tracks.items() 
                          if track['age'] < self.max_age}
            for track in self.tracks.values():
                track['age'] += 1
            return []
        
        # Match detections to existing tracks
        matched_tracks = []
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = det['bbox']
                self.tracks[best_track_id]['confidence'] = det['confidence']
                self.tracks[best_track_id]['age'] = 0
                det['track_id'] = best_track_id
            else:
                # Create new track
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': 0
                }
                self.next_id += 1
            
            matched_tracks.append(det)
        
        # Age out unmatched tracks
        active_ids = {det['track_id'] for det in matched_tracks}
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        return matched_tracks
    
    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# ============================================================================
# PHASE 2 & 3: TEAM CLASSIFICATION (Simplified)
# ============================================================================

class TeamClassifier:
    """
    Simplified team classifier using spatial information  
    (Full SigLIP clustering can be added later)
    """
    
    def __init__(self, net_x_image=None):
        """
        Initialize team classifier
        
        Args:
            net_x_image: X-coordinate of net in image space
        """
        self.net_x = net_x_image
        self.team_assignments = {}  # track_id -> team_id
    
    def set_net_position(self, net_x):
        """Set net x-coordinate in image"""
        self.net_x = net_x
    
    def classify_players(self, players: List[dict]):
        """
        Classify players into teams based on net side
        
        Args:
            players: List of player detections with track_ids
        
        Returns:
            players with 'team_id' added
        """
        if self.net_x is None:
            # Default: assume net at image center
            self.net_x = 640  # Typical HD width / 2
        
        for player in players:
            # Get player x-position (center of bbox)
            bbox = player['bbox']
            player_x = (bbox[0] + bbox[2]) / 2
            
            # Assign team based on side of net
            if player_x < self.net_x:
                player['team_id'] = 0  # Team A (left)
            else:
                player['team_id'] = 1  # Team B (right)
            
            # Cache assignment
            if 'track_id' in player:
                self.team_assignments[player['track_id']] = player['team_id']
        
        return players

# ============================================================================
# PHASE 4: COURT DETECTION (Simplified)
# ============================================================================

def detect_court_region(frame: np.ndarray, method='manual'):
    """
    Detect court region in frame
    
    Args:
        frame: Input frame
        method: 'manual' or 'auto'
    
    Returns:
        corners: 4 court corners in image coordinates
    """
    h, w = frame.shape[:2]
    
    if method == 'manual':
        # Use typical broadcast camera view
        # Court usually occupies central 70-80% of frame
        margin_x = int(w * 0.15)
        margin_y_top = int(h * 0.20)
        margin_y_bottom = int(h * 0.05)
        
        corners = np.array([
            [margin_x, margin_y_top],           # Top-left
            [w - margin_x, margin_y_top],       # Top-right
            [w - margin_x, h - margin_y_bottom],  # Bottom-right
            [margin_x, h - margin_y_bottom]     # Bottom-left
        ], dtype=np.float32)
        
        return corners
    
    # Auto-detection can be added here using Hough lines
    return None

# ============================================================================
# PHASE 5: VIEW TRANSFORMATION
# ============================================================================

class ViewTransformer:
    """Transform between image view and court view using homography"""
    
    def __init__(self, court_config: VolleyballCourtConfiguration):
        self.court_config = court_config
        self.homography_matrix = None
        self.inverse_homography = None
    
    def compute_homography(self, image_corners: np.ndarray):
        """
        Compute homography matrix
        
        Args:
            image_corners: 4 corners in image coordinates
        """
        court_corners = self.court_config.VERTICES
        
        self.homography_matrix = cv2.getPerspectiveTransform(
            image_corners, court_corners
        )
        self.inverse_homography = cv2.getPerspectiveTransform(
            court_corners, image_corners
        )
        
        logger.info("Homography matrix computed")
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform point from image to court coordinates"""
        if self.homography_matrix is None:
            return point
        
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.homography_matrix)
        
        x, y = transformed[0][0]
        
        # Clamp to court bounds
        x = np.clip(x, 0, self.court_config.COURT_LENGTH)
        y = np.clip(y, 0, self.court_config.COURT_WIDTH)
        
        return (float(x), float(y))
    
    def transform_players(self, players: List[dict]) -> List[dict]:
        """Transform player positions to court coordinates"""
        for player in players:
            bbox = player['bbox']
            # Use bottom-center of bounding box (feet position)
            bottom_center = ((bbox[0] + bbox[2]) / 2, bbox[3])
            court_pos = self.transform_point(bottom_center)
            player['court_position'] = court_pos
        
        return players
    
    def transform_ball(self, ball: dict) -> dict:
        """Transform ball position to court coordinates"""
        bbox = ball['bbox']
        # Use center of bounding box
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        court_pos = self.transform_point(center)
        ball['court_position'] = court_pos
        return ball

# ============================================================================
# PHASE 6: 2D TACTICAL MAP RENDERING
# ============================================================================

def render_tactical_map(players: List[dict], 
                       ball: Optional[dict],
                       ball_trajectory: List[Tuple],
                       court_img: np.ndarray) -> np.ndarray:
    """
    Render 2D tactical map with players and ball
    
    Args:
        players: List of players with court_position and team_id
        ball: Ball detection with court_position (or None)
        ball_trajectory: List of historical ball positions
        court_img: Base court image
    
    Returns:
        Rendered tactical map
    """
    map_img = court_img.copy()
    h, w = map_img.shape[:2]
    config = VolleyballCourtConfiguration
    
    # Scale factor for court coordinates to pixel coordinates
    scale_x = w / config.COURT_WIDTH
    scale_y = h / config.COURT_LENGTH
    
    # Draw ball trajectory
    if len(ball_trajectory) > 1:
        for i in range(len(ball_trajectory) - 1):
            pt1 = ball_trajectory[i]
            pt2 = ball_trajectory[i + 1]
            
            # Convert to pixel coordinates
            px1 = int(pt1[1] * scale_x)  # y -> x (rotated)
            py1 = int(pt1[0] * scale_y)  # x -> y
            px2 = int(pt2[1] * scale_x)
            py2 = int(pt2[0] * scale_y)
            
            # Fade older positions
            alpha = (i + 1) / len(ball_trajectory)
            color = tuple(int(c * alpha) for c in config.BALL_COLOR)
            
            cv2.line(map_img, (px1, py1), (px2, py2), color, thickness=2)
    
    # Draw players
    for player in players:
        if 'court_position' not in player:
            continue
        
        court_x, court_y = player['court_position']
        
        # Convert to pixel coordinates (note: court x,y != pixel x,y)
        px = int(court_y * scale_x)  # court_y maps to pixel_x
        py = int(court_x * scale_y)  # court_x maps to pixel_y
        
        # Get team color
        team_id = player.get('team_id', 0)
        color = config.TEAM_A_COLOR if team_id == 0 else config.TEAM_B_COLOR
        
        # Draw player dot
        cv2.circle(map_img, (px, py), radius=15, color=color, thickness=-1)
        cv2.circle(map_img, (px, py), radius=15, color=(255, 255, 255), thickness=2)
        
        # Draw track ID
        if 'track_id' in player:
            cv2.putText(map_img, str(player['track_id']), 
                       (px - 10, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 1)
    
    # Draw ball
    if ball is not None and 'court_position' in ball:
        court_x, court_y = ball['court_position']
        px = int(court_y * scale_x)
        py = int(court_x * scale_y)
        
        cv2.circle(map_img, (px, py), radius=10, 
                  color=config.BALL_COLOR, thickness=-1)
        cv2.circle(map_img, (px, py), radius=10, 
                  color=(0, 0, 0), thickness=2)
    
    return map_img

# ============================================================================
# PHASE 7: BALL TRAJECTORY & RALLY INTELLIGENCE
# ============================================================================

class BallTrajectory:
    """Track and smooth ball trajectory"""
    
    def __init__(self, max_length=30, smoothing_alpha=0.7):
        """
        Initialize ball trajectory tracker
        
        Args:
            max_length: Maximum trajectory length
            smoothing_alpha: EMA smoothing factor (0-1)
        """
        self.positions = deque(maxlen=max_length)
        self.alpha = smoothing_alpha
        self.smoothed_pos = None
        self.frames_since_detection = 0
    
    def update(self, ball_position: Optional[Tuple[float, float]]):
        """Update with new ball position"""
        if ball_position is not None:
            # Apply EMA smoothing
            if self.smoothed_pos is None:
                self.smoothed_pos = ball_position
            else:
                self.smoothed_pos = (
                    self.alpha * ball_position[0] + (1 - self.alpha) * self.smoothed_pos[0],
                    self.alpha * ball_position[1] + (1 - self.alpha) * self.smoothed_pos[1]
                )
            
            self.positions.append(self.smoothed_pos)
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Get trajectory positions"""
        return list(self.positions)
    
    def should_reset(self) -> bool:
        """Check if trajectory should reset (rally ended)"""
        # Reset if ball not detected for 30 frames
        return self.frames_since_detection > 30
    
    def reset(self):
        """Reset trajectory"""
        self.positions.clear()
        self.smoothed_pos = None
        self.frames_since_detection = 0

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_volleyball_video(
    video_path: str,
    weights_path: str,
    output_path: str = "output_tactical.mp4",
    device: str = 'cuda'
):
    """
    Main pipeline for volleyball tactical analysis
    
    Args:
        video_path: Path to input video
        weights_path: Path to RT-DETR weights
        output_path: Path for output video
        device: 'cuda' or 'cpu'
    """
    logger.info("="*60)
    logger.info("VOLLEYBALL HAWK-EYE TACTICAL INTELLIGENCE SYSTEM")
    logger.info("="*60)
    
    # Initialize components
    logger.info("\n[Phase 1] Initializing detector and tracker...")
    detector = RTDETRDetector(weights_path, device=device)
    player_tracker = SimpleTracker(max_age=30)
    ball_tracker = SimpleTracker(max_age=5)
    
    logger.info("[Phase 2-3] Initializing team classifier...")
    team_classifier = TeamClassifier()
    
    logger.info("[Phase 0] Creating court model...")
    court_config = VolleyballCourtConfiguration()
    court_base = draw_volleyball_pitch(width=450, height=900)
    
    logger.info("[Phase 5] Initializing view transformer...")
    transformer = ViewTransformer(court_config)
    
    logger.info("[Phase 7] Initializing ball trajectory tracker...")
    trajectory = BallTrajectory(max_length=50)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"\nVideo properties:")
    logger.info(f"  Resolution: {frame_width}x{frame_height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Total frames: {total_frames}")
    
    # Create video writer (side-by-side layout)
    output_width = frame_width + court_base.shape[1]
    output_height = max(frame_height, court_base.shape[0])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                          (output_width, output_height))
    
    # [Phase 4] Compute homography (first frame)
    ret, first_frame = cap.read()
    if ret:
        court_corners = detect_court_region(first_frame)
        transformer.compute_homography(court_corners)
        
        # Estimate net position (center of frame for now)
        net_x_estimate = frame_width // 2
        team_classifier.set_net_position(net_x_estimate)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    logger.info(f"\n[Phase 6] Processing video...")
    logger.info(f"Output: {output_path}")
    logger.info("")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # [Phase 1] Detect objects
            detections = detector.detect(frame, conf_threshold=0.25)
            
            # Track players
            tracked_players = player_tracker.update(detections['players'])
            
            # Track ball
            tracked_balls = ball_tracker.update(detections['balls'])
            ball = tracked_balls[0] if tracked_balls else None
            
            # [Phase 2-3] Classify teams
            tracked_players = team_classifier.classify_players(tracked_players)
            
            # [Phase 5] Transform to court coordinates
            tracked_players = transformer.transform_players(tracked_players)
            if ball is not None:
                ball = transformer.transform_ball(ball)
            
            # [Phase 7] Update ball trajectory
            ball_pos = ball['court_position'] if ball is not None else None
            trajectory.update(ball_pos)
            
            if trajectory.should_reset():
                trajectory.reset()
            
            # Draw detections on frame
            for player in tracked_players:
                bbox = player['bbox']
                team_id = player.get('team_id', 0)
                track_id = player.get('track_id', -1)
                
                color = court_config.TEAM_A_COLOR if team_id == 0 else court_config.TEAM_B_COLOR
                
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color, 2)
                
                label = f"P{track_id} T{team_id}"
                cv2.putText(annotated_frame, label, 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if ball is not None:
                bbox = ball['bbox']
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            court_config.BALL_COLOR, 2)
                cv2.putText(annotated_frame, "BALL", 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           court_config.BALL_COLOR, 2)
            
            # [Phase 6] Render tactical map
            tactical_map = render_tactical_map(
                tracked_players, 
                ball, 
                trajectory.get_positions(),
                court_base
            )
            
            # Resize tactical map to match frame height
            if tactical_map.shape[0] != frame_height:
                tactical_map = cv2.resize(tactical_map, 
                                         (tactical_map.shape[1], frame_height))
            
            # Combine frame and tactical map side-by-side
            combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            combined[:frame_height, :frame_width] = annotated_frame
            combined[:tactical_map.shape[0], frame_width:frame_width+tactical_map.shape[1]] = tactical_map
            
            # Write frame
            out.write(combined)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nError during processing: {e}")
        raise
    finally:
        cap.release()
        out.release()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete!")
        logger.info(f"Frames processed: {frame_count}/{total_frames}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"{'='*60}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    WEIGHTS_PATH = r"C:\Users\xghostrider\Downloads\best(2).pt"
    VIDEO_PATH = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\input_videos\Video1.mp4"
    OUTPUT_PATH = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\output_tactical.mp4"
    
    # Check if paths exist
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"Weights file not found: {WEIGHTS_PATH}")
        exit(1)
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f"Video file not found: {VIDEO_PATH}")
        exit(1)
    
    # Run pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    process_volleyball_video(
        video_path=VIDEO_PATH,
        weights_path=WEIGHTS_PATH,
        output_path=OUTPUT_PATH,
        device=device
    )
