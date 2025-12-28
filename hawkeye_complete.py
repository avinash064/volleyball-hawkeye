"""
Volleyball Hawk-Eye Tactical Intelligence System
Production-ready pipeline for volleyball tactical analysis

Engineering Principles:
- Robust heuristics over fragile perfection
- Volleyball-specific rules enforced
- Fail-safe design (no crashes on missing detections)
- Single integrated pipeline
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 0: VOLLEYBALL COURT MODEL
# ============================================================================

@dataclass
class VolleyballCourtConfiguration:
    """Official FIVB volleyball court specifications"""
    
    # Dimensions in centimeters
    COURT_LENGTH: int = 1800  # 18 meters
    COURT_WIDTH: int = 900    # 9 meters
    NET_X: int = 900          # Net at center
    
    # Attack lines (3m from net)
    ATTACK_LINE_DISTANCE: int = 300
    
    # Court vertices for homography
    VERTICES = np.array([
        [0, 0],                      # Top-left
        [COURT_LENGTH, 0],           # Top-right
        [COURT_LENGTH, COURT_WIDTH], # Bottom-right
        [0, COURT_WIDTH]             # Bottom-left
    ], dtype=np.float32)
    
    # Visualization colors (BGR)
    COURT_COLOR = (139, 69, 19)     # Brown
    LINE_COLOR = (255, 255, 255)    # White
    NET_COLOR = (200, 200, 200)     # Light gray
    
    TEAM_A_COLOR = (255, 100, 100)  # Light blue
    TEAM_B_COLOR = (100, 100, 255)  # Light red
    BALL_COLOR = (0, 255, 255)      # Yellow
    REFEREE_COLOR = (128, 128, 128) # Gray

def draw_volleyball_court(width=450, height=900):
    """
    Draw top-down 2D volleyball court
    
    Args:
        width: Canvas width (represents 9m court width)
        height: Canvas height (represents 18m court length)
    
    Returns:
        court_img: Rendered court (BGR)
    """
    config = VolleyballCourtConfiguration()
    court_img = np.full((height, width, 3), config.COURT_COLOR, dtype=np.uint8)
    
    # Court outline
    cv2.rectangle(court_img, (0, 0), (width-1, height-1), 
                  config.LINE_COLOR, thickness=3)
    
    # Center line (net)
    net_y = height // 2
    cv2.line(court_img, (0, net_y), (width, net_y), 
             config.NET_COLOR, thickness=5)
    
    # Attack lines (3m from net)
    attack_offset = int(height * 0.167)  # 3m / 18m
    cv2.line(court_img, (0, net_y - attack_offset), 
             (width, net_y - attack_offset), 
             config.LINE_COLOR, thickness=2)
    cv2.line(court_img, (0, net_y + attack_offset), 
             (width, net_y + attack_offset), 
             config.LINE_COLOR, thickness=2)
    
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(court_img, "TEAM A", (int(width*0.32), int(height*0.25)), 
                font, 1.2, config.LINE_COLOR, 2)
    cv2.putText(court_img, "TEAM B", (int(width*0.32), int(height*0.75)), 
                font, 1.2, config.LINE_COLOR, 2)
    cv2.putText(court_img, "NET", (int(width*0.38), net_y - 10), 
                font, 0.8, config.LINE_COLOR, 2)
    
    return court_img

# ============================================================================
# PHASE 1: DETECTION & TRACKING
# ============================================================================

class RTDETRDetector:
    """RT-DETR object detector wrapper"""
    
    def __init__(self, weights_path: str, device='cuda', conf_threshold=0.25):
        """Initialize RT-DETR"""
        self.device = device
        self.conf_threshold = conf_threshold
        logger.info(f"Loading RT-DETR: {weights_path}")
        
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(weights_path)
            self.model.to(device)
            logger.info("RT-DETR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RT-DETR: {e}")
            raise
    
    def detect(self, frame: np.ndarray):
        """
        Run detection
        
        Returns:
            Dict with 'players', 'referees', 'balls'
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = {'players': [], 'referees': [], 'balls': []}
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            class_names = results.names
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                class_name = class_names[cls].lower()
                
                det = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name
                }
                
                if 'player' in class_name:
                    detections['players'].append(det)
                elif 'referee' in class_name:
                    detections['referees'].append(det)
                elif 'volleyball' in class_name or 'ball' in class_name:
                    detections['balls'].append(det)
        
        return detections

class ByteTracker:
    """
    Simple ByteTrack-style tracker using IoU matching
    Production-ready with occlusion handling
    """
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
    
    def update(self, detections: List[dict]) -> List[dict]:
        """Update tracks with new detections"""
        if not detections:
            # Age out old tracks
            for track in list(self.tracks.values()):
                track['age'] += 1
            self.tracks = {tid: t for tid, t in self.tracks.items() 
                          if t['age'] < self.max_age}
            return []
        
        matched = []
        
        # Match detections to tracks
        for det in detections:
            best_iou = 0
            best_id = None
            
            for tid, track in self.tracks.items():
                iou = self._iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                # Update existing track
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['confidence'] = det['confidence']
                self.tracks[best_id]['age'] = 0
                det['track_id'] = best_id
            else:
                # New track
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': 0
                }
                self.next_id += 1
            
            matched.append(det)
        
        # Age unmatched tracks
        active_ids = {d['track_id'] for d in matched}
        for tid in self.tracks:
            if tid not in active_ids:
                self.tracks[tid]['age'] += 1
        
        return matched
    
    @staticmethod
    def _iou(box1, box2):
        """Compute IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

# ============================================================================
# PHASE 2: TEAM CLASSIFICATION (SigLIP + Spatial)
# ============================================================================

class TeamClassifier:
    """
    Team classification using:
    1. SigLIP embeddings (jersey appearance)
    2. Spatial constraints (net-side validation)
    """
    
    def __init__(self, net_x=None, use_siglip=False):
        """
        Args:
            net_x: Net x-coordinate in image
            use_siglip: Whether to use SigLIP (requires transformers)
        """
        self.net_x = net_x
        self.use_siglip = use_siglip
        self.team_cache = {}  # track_id -> team_id
        self.embeddings = {}  # track_id -> embedding
        self.trained = False
        
        if use_siglip:
            try:
                from transformers import AutoModel, AutoProcessor
                self.siglip_model = AutoModel.from_pretrained(
                    "google/siglip-base-patch16-224"
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                self.siglip_processor = AutoProcessor.from_pretrained(
                    "google/siglip-base-patch16-224"
                )
                self.siglip_model.eval()
                logger.info("SigLIP loaded for team classification")
            except Exception as e:
                logger.warning(f"SigLIP loading failed: {e}. Using spatial only.")
                self.use_siglip = False
    
    def set_net_position(self, net_x):
        """Set net x-coordinate"""
        self.net_x = net_x
    
    def extract_embedding(self, frame, bbox):
        """Extract SigLIP embedding from player crop"""
        if not self.use_siglip:
            return None
        
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inputs = self.siglip_processor(images=crop_rgb, return_tensors="pt")
            inputs = {k: v.to(self.siglip_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.siglip_model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None
    
    def train_clustering(self, embeddings_list, n_clusters=2):
        """
        Train KMeans clustering on collected embeddings
        
        Args:
            embeddings_list: List of (track_id, embedding) tuples
            n_clusters: Number of teams (always 2 for volleyball)
        """
        if len(embeddings_list) < 4:  # Need minimum samples
            return False
        
        try:
            from sklearn.cluster import KMeans
            
            track_ids = [tid for tid, _ in embeddings_list]
            embeddings = np.array([emb for _, emb in embeddings_list])
            
            # Cluster into 2 teams
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Update cache
            for tid, label in zip(track_ids, labels):
                self.team_cache[tid] = int(label)
            
            self.trained = True
            logger.info(f"Team clustering trained on {len(embeddings_list)} samples")
            return True
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return False
    
    def classify_players(self, players: List[dict], frame=None):
        """
        Classify players into teams
        
        Uses SigLIP if available and trained, otherwise spatial
        """
        if self.net_x is None:
            self.net_x = 640  # Default
        
        for player in players:
            track_id = player.get('track_id')
            bbox = player['bbox']
            player_x = (bbox[0] + bbox[2]) / 2
            
            # Try cached team first
            if track_id in self.team_cache:
                player['team_id'] = self.team_cache[track_id]
            else:
                # Fallback to spatial classification
                team_id = 0 if player_x < self.net_x else 1
                player['team_id'] = team_id
                
                if track_id is not None:
                    self.team_cache[track_id] = team_id
            
            # Extract embeddings for future training (if SigLIP enabled)
            if self.use_siglip and frame is not None and track_id is not None:
                if track_id not in self.embeddings:
                    emb = self.extract_embedding(frame, bbox)
                    if emb is not None:
                        self.embeddings[track_id] = emb
        
        return players

# ============================================================================
# PHASE 4: COURT DETECTION (Robust Heuristic)
# ============================================================================

def detect_court_region(frame: np.ndarray):
    """
    Robust court detection using broadcast camera heuristics
    
    Volleyball broadcasts typically have:
    - Court centered in frame
    - ~70-80% of frame height
    - ~70-85% of frame width
    
    Returns:
        corners: 4 court corners [TL, TR, BR, BL]
    """
    h, w = frame.shape[:2]
    
    # Conservative margins for volleyball broadcast
    margin_x = int(w * 0.12)
    margin_y_top = int(h * 0.18)
    margin_y_bottom = int(h * 0.08)
    
    corners = np.array([
        [margin_x, margin_y_top],                  # Top-left
        [w - margin_x, margin_y_top],              # Top-right
        [w - margin_x, h - margin_y_bottom],       # Bottom-right
        [margin_x, h - margin_y_bottom]            # Bottom-left
    ], dtype=np.float32)
    
    return corners

# ============================================================================
# PHASE 5: VIEW TRANSFORMATION
# ============================================================================

class ViewTransformer:
    """Perspective transformation: image ← → court"""
    
    def __init__(self, court_config: VolleyballCourtConfiguration):
        self.court_config = court_config
        self.H = None  # Image → Court
        self.H_inv = None  # Court → Image
    
    def compute_homography(self, image_corners: np.ndarray):
        """Compute homography from detected court corners"""
        court_corners = self.court_config.VERTICES
        
        self.H = cv2.getPerspectiveTransform(image_corners, court_corners)
        self.H_inv = cv2.getPerspectiveTransform(court_corners, image_corners)
        
        logger.info("Homography matrix computed")
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform point from image to court coordinates"""
        if self.H is None:
            return point
        
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        
        x, y = transformed[0][0]
        
        # Clamp to court bounds (CRITICAL for volleyball)
        x = np.clip(x, 0, self.court_config.COURT_LENGTH)
        y = np.clip(y, 0, self.court_config.COURT_WIDTH)
        
        return (float(x), float(y))
    
    def transform_players(self, players: List[dict]) -> List[dict]:
        """Transform players using bottom-center (feet)"""
        for player in players:
            bbox = player['bbox']
            bottom_center = ((bbox[0] + bbox[2]) / 2, bbox[3])
            player['court_position'] = self.transform_point(bottom_center)
        return players
    
    def transform_ball(self, ball: dict) -> dict:
        """Transform ball using center"""
        bbox = ball['bbox']
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        ball['court_position'] = self.transform_point(center)
        return ball

# ============================================================================
# PHASE 7: TRAJECTORIES & BALL INTELLIGENCE
# ============================================================================

class BallTrajectory:
    """Ball trajectory tracker with EMA smoothing"""
    
    def __init__(self, max_length=50, alpha=0.7):
        self.positions = deque(maxlen=max_length)
        self.alpha = alpha
        self.smoothed = None
        self.frames_missing = 0
    
    def update(self, ball_pos: Optional[Tuple[float, float]]):
        """Update trajectory"""
        if ball_pos is not None:
            if self.smoothed is None:
                self.smoothed = ball_pos
            else:
                self.smoothed = (
                    self.alpha * ball_pos[0] + (1 - self.alpha) * self.smoothed[0],
                    self.alpha * ball_pos[1] + (1 - self.alpha) * self.smoothed[1]
                )
            self.positions.append(self.smoothed)
            self.frames_missing = 0
        else:
            self.frames_missing += 1
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory points"""
        return list(self.positions)
    
    def should_reset(self) -> bool:
        """Reset if ball missing too long (rally ended)"""
        return self.frames_missing > 30
    
    def reset(self):
        """Reset trajectory"""
        self.positions.clear()
        self.smoothed = None
        self.frames_missing = 0

class PlayerMovementTracker:
    """Track player movement paths"""
    
    def __init__(self, max_trail=30):
        self.trails = defaultdict(lambda: deque(maxlen=max_trail))
    
    def update(self, players: List[dict]):
        """Update movement trails"""
        for player in players:
            if 'court_position' in player and 'track_id' in player:
                tid = player['track_id']
                self.trails[tid].append(player['court_position'])
    
    def get_trail(self, track_id) -> List[Tuple[float, float]]:
        """Get movement trail for player"""
        return list(self.trails.get(track_id, []))

# ============================================================================
# PHASE 6: TACTICAL MAP RENDERING
# ============================================================================

def render_tactical_map(
    players: List[dict],
    ball: Optional[dict],
    ball_trajectory: List[Tuple],
    player_trails: PlayerMovementTracker,
    court_img: np.ndarray
) -> np.ndarray:
    """
    Render 2D tactical map with:
    - Court layout
    - Player positions (team-colored)
    - Player movement trails
    - Ball position
    - Ball trajectory arc
    """
    map_img = court_img.copy()
    h, w = map_img.shape[:2]
    config = VolleyballCourtConfiguration()
    
    scale_x = w / config.COURT_WIDTH
    scale_y = h / config.COURT_LENGTH
    
    def court_to_pixel(court_pos):
        """Convert court coordinates to pixel coordinates"""
        x, y = court_pos
        px = int(y * scale_x)  # court_y → pixel_x (rotated)
        py = int(x * scale_y)  # court_x → pixel_y
        return (px, py)
    
    # Draw ball trajectory
    if len(ball_trajectory) > 1:
        for i in range(len(ball_trajectory) - 1):
            pt1 = court_to_pixel(ball_trajectory[i])
            pt2 = court_to_pixel(ball_trajectory[i + 1])
            
            alpha = (i + 1) / len(ball_trajectory)
            color = tuple(int(c * alpha) for c in config.BALL_COLOR)
            
            cv2.line(map_img, pt1, pt2, color, thickness=2)
    
    # Draw player trails
    for player in players:
        if 'track_id' not in player:
            continue
        
        trail = player_trails.get_trail(player['track_id'])
        if len(trail) > 1:
            team_id = player.get('team_id', 0)
            color = config.TEAM_A_COLOR if team_id == 0 else config.TEAM_B_COLOR
            
            for i in range(len(trail) - 1):
                pt1 = court_to_pixel(trail[i])
                pt2 = court_to_pixel(trail[i + 1])
                alpha = (i + 1) / len(trail)
                trail_color = tuple(int(c * alpha * 0.6) for c in color)
                cv2.line(map_img, pt1, pt2, trail_color, thickness=2)
    
    # Draw players
    for player in players:
        if 'court_position' not in player:
            continue
        
        px, py = court_to_pixel(player['court_position'])
        team_id = player.get('team_id', 0)
        color = config.TEAM_A_COLOR if team_id == 0 else config.TEAM_B_COLOR
        
        cv2.circle(map_img, (px, py), radius=12, color=color, thickness=-1)
        cv2.circle(map_img, (px, py), radius=12, color=(255, 255, 255), thickness=2)
        
        if 'track_id' in player:
            cv2.putText(map_img, str(player['track_id']), 
                       (px - 8, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw ball
    if ball and 'court_position' in ball:
        px, py = court_to_pixel(ball['court_position'])
        cv2.circle(map_img, (px, py), radius=8, 
                  color=config.BALL_COLOR, thickness=-1)
        cv2.circle(map_img, (px, py), radius=8, 
                  color=(0, 0, 0), thickness=2)
    
    return map_img

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def volleyball_hawkeye_pipeline(
    video_path: str,
    weights_path: str,
    output_path: str,
    device='cuda',
    use_siglip=False
):
    """
    Complete Volleyball Hawk-Eye pipeline
    
    Args:
        video_path: Input video path
        weights_path: RT-DETR weights
        output_path: Output video path
        device: 'cuda' or 'cpu'
        use_siglip: Enable SigLIP team classification
    """
    logger.info("="*70)
    logger.info("VOLLEYBALL HAWK-EYE TACTICAL INTELLIGENCE SYSTEM")
    logger.info("Production-Ready Pipeline")
    logger.info("="*70)
    
    # Phase 1: Detection & Tracking
    logger.info("\n[Phase 1] Initializing detection & tracking...")
    detector = RTDETRDetector(weights_path, device=device)
    player_tracker = ByteTracker(max_age=30)
    ball_tracker = ByteTracker(max_age=5)
    
    # Phase 2: Team Classification
    logger.info("[Phase 2] Initializing team classifier...")
    team_classifier = TeamClassifier(use_siglip=use_siglip)
    
    # Phase 0: Court Model
    logger.info("[Phase 0] Creating court model...")
    court_config = VolleyballCourtConfiguration()
    court_base = draw_volleyball_court(width=450, height=900)
    
    # Phase 5: View Transformer
    logger.info("[Phase 5] Initializing view transformer...")
    transformer = ViewTransformer(court_config)
    
    # Phase 7: Trajectories
    logger.info("[Phase 7] Initializing trajectory trackers...")
    ball_trajectory = BallTrajectory(max_length=50)
    player_movement = PlayerMovementTracker(max_trail=30)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"\nVideo: {frame_width}x{frame_height} @ {fps} FPS")
    logger.info(f"Total frames: {total_frames}")
    
    # Video writer
    output_width = frame_width + court_base.shape[1]
    output_height = max(frame_height, court_base.shape[0])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Phase 4: Compute homography (first frame)
    ret, first_frame = cap.read()
    if ret:
        court_corners = detect_court_region(first_frame)
        transformer.compute_homography(court_corners)
        
        net_x = frame_width // 2
        team_classifier.set_net_position(net_x)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    logger.info("\n[Phase 6] Processing video...\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Detect
            detections = detector.detect(frame)
            
            # Track
            tracked_players = player_tracker.update(detections['players'])
            tracked_balls = ball_tracker.update(detections['balls'])
            ball = tracked_balls[0] if tracked_balls else None
            
            # Classify teams
            tracked_players = team_classifier.classify_players(tracked_players, frame)
            
            # Transform to court
            tracked_players = transformer.transform_players(tracked_players)
            if ball:
                ball = transformer.transform_ball(ball)
            
            # Update trajectories
            ball_pos = ball['court_position'] if ball else None
            ball_trajectory.update(ball_pos)
            
            if ball_trajectory.should_reset():
                ball_trajectory.reset()
            
            player_movement.update(tracked_players)
            
            # Render broadcast view
            broadcast_frame = frame.copy()
            
            for player in tracked_players:
                bbox = player['bbox']
                team_id = player.get('team_id', 0)
                track_id = player.get('track_id', -1)
                
                color = court_config.TEAM_A_COLOR if team_id == 0 else court_config.TEAM_B_COLOR
                
                cv2.rectangle(broadcast_frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color, 3)
                
                label = f"P{track_id} T{team_id}"
                cv2.putText(broadcast_frame, label, 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if ball:
                bbox = ball['bbox']
                cv2.rectangle(broadcast_frame, 
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            court_config.BALL_COLOR, 3)
                cv2.putText(broadcast_frame, "BALL", 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           court_config.BALL_COLOR, 2)
            
            # Render tactical map
            tactical_map = render_tactical_map(
                tracked_players,
                ball,
                ball_trajectory.get_trajectory(),
                player_movement,
                court_base
            )
            
            # Resize tactical map
            if tactical_map.shape[0] != frame_height:
                tactical_map = cv2.resize(tactical_map, 
                                         (tactical_map.shape[1], frame_height))
            
            # Combine
            combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            combined[:frame_height, :frame_width] = broadcast_frame
            combined[:tactical_map.shape[0], 
                    frame_width:frame_width+tactical_map.shape[1]] = tactical_map
            
            out.write(combined)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        cap.release()
        out.release()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Complete! Processed {frame_count}/{total_frames} frames")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'='*70}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    WEIGHTS = r"C:\Users\xghostrider\Downloads\best(2).pt"
    VIDEO = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\input_videos\Video1.mp4"
    OUTPUT = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\hawkeye_complete.mp4"
    
    if not Path(WEIGHTS).exists():
        logger.error(f"Weights not found: {WEIGHTS}")
        exit(1)
    
    if not Path(VIDEO).exists():
        logger.error(f"Video not found: {VIDEO}")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    volleyball_hawkeye_pipeline(
        video_path=VIDEO,
        weights_path=WEIGHTS,
        output_path=OUTPUT,
        device=device,
        use_siglip=False  # Set True to enable SigLIP (requires transformers)
    )
