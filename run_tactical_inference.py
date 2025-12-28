"""
Memory-efficient volleyball inference with tactical court view
Resizes frames to prevent memory issues with high-res videos
"""

import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path
from detection import VolleyballDetector
from tracking import ByteTrack, TeamClassifier
from projection import CourtHomography
from tqdm import tqdm

def main():
    print("=" * 60)
    print("Volleyball Inference - 2D Tactical Court View")
    print("=" * 60)
    
    # Configuration
    video_path = "input_videos/Video2.mp4"
    output_path = "outputs/tactical_view.mp4"
    max_width = 960  # Smaller resize for 4GB GPU
    
    #Initialize
    print("\n[1/5] Loading video...")
    cap = cv2.VideoCapture(video_path)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate resize scale
    scale = max_width / original_w
    new_w = max_width
    new_h = int(original_h * scale)
    
    print(f"  Original: {original_w}x{original_h}")
    print(f"  Resized: {new_w}x{new_h}")
    print(f"  FPS: {fps}, Frames: {total_frames}")
    
    print("\n[2/5] Initializing detector...")
    detector = VolleyballDetector("yolov8m.pt", device="cpu",
                                 conf_player=0.35, conf_ball=0.15)
    
    print("\n[3/5] Initializing tracker and classifier...")
    tracker = ByteTrack(frame_rate=fps)
    team_classifier = TeamClassifier()
    
    print("\n[4/5] Initializing court projection...")
    court_proj = CourtHomography()
    court_proj.load_keypoints("configs/court_keypoints.json")
    
    # Output setup
    Path("outputs").mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create side-by-side output (main view + tactical view)
    tactical_w = 400
    tactical_h = 600
    out_w = new_w + tactical_w
    out_h = max(new_h, tactical_h)
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    print(f"\n[5/5] Processing {total_frames} frames...")
    print("=" * 60)
    
    frame_count = 0
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to save memory
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Detection
        detections = detector.detect(frame)
        
        # Tracking
        tracks = tracker.update(detections)
        
        # Team classification (skip for now - just use random team assignment)
        # tracks = team_classifier.classify_teams(tracks, frame)
        
        # Draw on main view
        output_frame = frame.copy()
        
        for idx, track in enumerate(tracks):
            x1, y1, x2, y2 = track['bbox']
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = track['track_id']
            
            # Simple team assignment based on ID (temporary)
            team_id = track_id % 2
            team = f"Team {'A' if team_id == 0 else 'B'}"
            
            # Color based on team
            color = (0, 255, 0) if team_id == 0 else (0, 0, 255)  # Green/Red
            
            # Draw bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {team}"
            cv2.putText(output_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Create tactical view (empty court)
        tactical_view = court_proj.visualize_court(np.array([]), output_size=(tactical_w, tactical_h))
        
        # Project players onto tactical view
        for idx, track in enumerate(tracks):
            # Get foot point (bottom-center of bbox)
            x1, y1, x2, y2 = track['bbox']
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            
            court_coords = court_proj.project_single(foot_x, foot_y)
            if court_coords is not None:
                team_id = track['track_id'] % 2
                color = (0, 255, 0) if team_id == 0 else (0, 0, 255)
                
                # Draw on tactical view
                x, y = court_coords
                px = int(tactical_w / 2 + x * 40)  # Scale to tactical view
                py = int(tactical_h - 50 - y * 50)
                
                if 0 <= px < tactical_w and 0 <= py < tactical_h:
                    cv2.circle(tactical_view, (px, py), 8, color, -1)
                    cv2.putText(tactical_view, str(track['track_id']),
                               (px - 10, py - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine views
        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:new_h, :new_w] = output_frame
        combined[:tactical_h, new_w:] = tactical_view
        
        # Add labels
        cv2.putText(combined, "Main View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Tactical View (2D Court)", (new_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(combined)
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    writer.release()
    
    print("\n" + "=" * 60)
    print(f"[OK] Inference complete!")
    print(f"  Processed: {frame_count} frames")
    print(f"  Output: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
