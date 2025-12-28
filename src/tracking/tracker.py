"""
ByteTrack Multi-Object Tracker

Implements ByteTrack for robust tracking of volleyball players and ball
with identity persistence across frames.

Reference: https://arxiv.org/abs/2110.06864
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, deque


class Track:
    """Represents a single tracked object"""
    
    _id_counter = 0
    
    def __init__(self, detection: Dict, frame_id: int):
        """
        Initialize a new track.
        
        Args:
            detection: Initial detection dict with bbox, conf, class
            frame_id: Frame number where track starts
        """
        self.track_id = Track._id_counter
        Track._id_counter += 1
        
        self.bbox = detection["bbox"]
        self.conf = detection["conf"]
        self.class_name = detection["class"]
        self.frame_id = frame_id
        self.last_update = frame_id
        self.age = 0
        self.hits = 1
        
        # Trajectory history
        self.trajectory = deque(maxlen=30)
        self.trajectory.append(self._get_center())
    
    def update(self, detection: Dict, frame_id: int):
        """Update track with new detection"""
        self.bbox = detection["bbox"]
        self.conf = detection["conf"]
        self.last_update = frame_id
        self.hits += 1
        self.age = 0
        
        self.trajectory.append(self._get_center())
    
    def predict(self):
        """Predict next position (simple constant velocity model)"""
        if len(self.trajectory) < 2:
            return self.bbox
        
        # Calculate velocity
        p1 = np.array(self.trajectory[-1])
        p2 = np.array(self.trajectory[-2])
        velocity = p1 - p2
        
        # Predict center
        predicted_center = p1 + velocity
        
        # Update bbox (keep same size)
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        
        predicted_bbox = [
            predicted_center[0] - w / 2,
            predicted_center[1] - h / 2,
            predicted_center[0] + w / 2,
            predicted_center[1] + h / 2
        ]
        
        return predicted_bbox
    
    def _get_center(self) -> Tuple[float, float]:
        """Get bbox center"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def to_dict(self) -> Dict:
        """Export track to dict"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "conf": self.conf,
            "class": self.class_name,
            "trajectory": list(self.trajectory)
        }


class ByteTrack:
    """
    ByteTrack: Multi-Object Tracking with Two-Stage Association
    
    Uses high and low confidence thresholds to improve tracking robustness.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        """
        Initialize ByteTrack.
        
        Args:
            track_thresh: High confidence threshold for first-stage matching
            track_buffer: Frames to keep lost tracks before deletion
            match_thresh: IoU threshold for matching
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []     # Recently lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
        
        Returns:
            List of active tracks with IDs
        """
        self.frame_id += 1
        
        # Split detections by confidence
        high_conf_dets = [d for d in detections if d["conf"] >= self.track_thresh]
        low_conf_dets = [d for d in detections if d["conf"] < self.track_thresh]
        
        # Predict existing tracks
        for track in self.tracked_tracks:
            track.age += 1
        
        # First association: high-confidence detections with tracked tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_conf_dets
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracked_tracks[track_idx].update(high_conf_dets[det_idx], self.frame_id)
        
        # Second association: unmatched tracks with low-confidence detections
        unmatched_tracked = [self.tracked_tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_tracks_low, _ = self._associate(
            unmatched_tracked, low_conf_dets
        )
        
        # Update matched low-confidence tracks
        for track_idx, det_idx in matched_low:
            unmatched_tracked[track_idx].update(low_conf_dets[det_idx], self.frame_id)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks_low:
            track = unmatched_tracked[track_idx]
            self.lost_tracks.append(track)
        
        # Remove tracked tracks that were moved to lost
        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if t not in self.lost_tracks
        ]
        
        # Initialize new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets:
            new_track = Track(high_conf_dets[det_idx], self.frame_id)
            self.tracked_tracks.append(new_track)
        
        # Try to re-activate lost tracks with low-confidence detections
        # (omitted for brevity, can be added if needed)
        
        # Remove old lost tracks
        self.lost_tracks = [
            t for t in self.lost_tracks
            if self.frame_id - t.last_update <= self.track_buffer
        ]
        
        # Return active tracks
        return [t.to_dict() for t in self.tracked_tracks]
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.
        
        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of track indices
            unmatched_dets: List of detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._iou(track.bbox, det["bbox"])
        
        # Greedy matching (can be replaced with Hungarian algorithm)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find best match
            max_iou = 0
            best_match = None
            
            for t_idx in unmatched_tracks:
                for d_idx in unmatched_dets:
                    if iou_matrix[t_idx, d_idx] > max_iou:
                        max_iou = iou_matrix[t_idx, d_idx]
                        best_match = (t_idx, d_idx)
            
            # Check if match is above threshold
            if max_iou < self.match_thresh:
                break
            
            # Add match
            t_idx, d_idx = best_match
            matched.append((t_idx, d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(d_idx)
        
        return matched, unmatched_tracks, unmatched_dets
    
    @staticmethod
    def _iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        Track._id_counter = 0
