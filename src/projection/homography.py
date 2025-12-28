"""
Court Homography for Volleyball

Transforms player positions from camera view to 2D court coordinates
using perspective transformation (homography matrix).
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import json


class CourtHomography:
    """
    Compute and apply homography transformation for volleyball court projection.
    
    Converts pixel coordinates in the broadcast view to metric coordinates
    on a standardized 2D court representation.
    """
    
    # Official volleyball court dimensions (meters)
    COURT_WIDTH = 9.0
    COURT_LENGTH = 18.0
    
    def __init__(self, keypoints_path: Optional[str] = None):
        """
        Initialize court homography.
        
        Args:
            keypoints_path: Path to JSON file with court keypoints
        """
        self.H = None  # Homography matrix
        self.src_points = None  # Source points in image
        self.dst_points = None  # Destination points on court
        
        if keypoints_path:
            self.load_keypoints(keypoints_path)
    
    def calibrate(
        self,
        image_points: np.ndarray,
        court_points: Optional[np.ndarray] = None
    ):
        """
        Calibrate homography from image-court point correspondences.
        
        Args:
            image_points: Nx2 array of points in image (pixel coords)
            court_points: Nx2 array of points on court (meter coords)
                         If None, uses standard court keypoints
        """
        if court_points is None:
            # Use default court corners
            court_points = self._get_default_court_points()
        
        assert len(image_points) >= 4, "Need at least 4 point correspondences"
        assert len(image_points) == len(court_points), "Point arrays must match"
        
        self.src_points = image_points.astype(np.float32)
        self.dst_points = court_points.astype(np.float32)
        
        # Compute homography
        self.H, _ = cv2.findHomography(self.src_points, self.dst_points)
        
        print(f"✅ Homography calibrated with {len(image_points)} points")
    
    def project_points(self, image_points: np.ndarray) -> np.ndarray:
        """
        Project points from image to court coordinates.
        
        Args:
            image_points: Nx2 array of points in image
        
        Returns:
            Nx2 array of points on court (meters)
        """
        if self.H is None:
            raise ValueError("Homography not calibrated. Call calibrate() first.")
        
        if len(image_points) == 0:
            return np.array([])
        
        # Reshape for cv2.perspectiveTransform
        points = image_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Transform
        court_points = cv2.perspectiveTransform(points, self.H)
        
        return court_points.reshape(-1, 2)
    
    def project_single(self, x: float, y: float) -> Tuple[float, float]:
        """Project a single point"""
        court_point = self.project_points(np.array([[x, y]]))
        return tuple(court_point[0])
    
    def _get_default_court_points(self) -> np.ndarray:
        """
        Get standard volleyball court keypoints in metric coordinates.
        
        Layout (top view):
        
            0 ----------- 1
            |             |
            |             |
            2 ----net---- 3
            |             |
            |             |
            4 ----------- 5
        
        Returns:
            6x2 array of court points (meters)
        """
        w, l = self.COURT_WIDTH, self.COURT_LENGTH
        
        court_points = np.array([
            [0, 0],      # Top-left corner
            [w, 0],      # Top-right corner
            [0, l/2],    # Left net post
            [w, l/2],    # Right net post
            [0, l],      # Bottom-left corner
            [w, l]       # Bottom-right corner
        ], dtype=np.float32)
        
        return court_points
    
    def save_keypoints(self, path: str):
        """Save calibration keypoints to JSON"""
        if self.src_points is None or self.dst_points is None:
            raise ValueError("No keypoints to save")
        
        data = {
            "image_points": self.src_points.tolist(),
            "court_points": self.dst_points.tolist(),
            "court_dimensions": {
                "width": self.COURT_WIDTH,
                "length": self.COURT_LENGTH
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved keypoints to {path}")
    
    def load_keypoints(self, path: str):
        """Load calibration keypoints from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        image_points = np.array(data["image_points"], dtype=np.float32)
        court_points = np.array(data["court_points"], dtype=np.float32)
        
        self.calibrate(image_points, court_points)
        
        print(f"✅ Loaded keypoints from {path}")
    
    def visualize_court(
        self,
        player_positions: np.ndarray,
        output_size: Tuple[int, int] = (900, 1800),
        team_labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Visualize player positions on 2D court.
        
        Args:
            player_positions: Nx2 array of court positions (meters)
            output_size: Output image size (width, height)
            team_labels: Optional list of team labels for each player
        
        Returns:
            Court visualization image
        """
        w, h = output_size
        court_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        # Scale factor (pixels per meter)
        scale_x = w / self.COURT_WIDTH
        scale_y = h / self.COURT_LENGTH
        
        # Draw court lines
        line_color = (100, 100, 100)
        line_thickness = 2
        
        # Boundary
        cv2.rectangle(
            court_img,
            (0, 0),
            (w - 1, h - 1),
            line_color,
            line_thickness
        )
        
        # Net (center line)
        net_y = int(h / 2)
        cv2.line(
            court_img,
            (0, net_y),
            (w, net_y),
            line_color,
            line_thickness * 2
        )
        
        # Attack lines (3m from net)
        attack_line_dist = 3.0  # meters
        attack_y1 = int((self.COURT_LENGTH / 2 - attack_line_dist) * scale_y)
        attack_y2 = int((self.COURT_LENGTH / 2 + attack_line_dist) * scale_y)
        
        cv2.line(court_img, (0, attack_y1), (w, attack_y1), line_color, 1)
        cv2.line(court_img, (0, attack_y2), (w, attack_y2), line_color, 1)
        
        # Draw players
        team_colors = {
            "Team A": (255, 100, 100),  # Light red
            "Team B": (100, 100, 255),  # Light blue
            "Referee": (100, 255, 100),  # Light green
            "Unknown": (150, 150, 150)  # Gray
        }
        
        for i, (x, y) in enumerate(player_positions):
            # Convert to pixel coords
            px = int(x * scale_x)
            py = int(y * scale_y)
            
            # Get team color
            if team_labels is not None and i < len(team_labels):
                color = team_colors.get(team_labels[i], team_colors["Unknown"])
            else:
                color = team_colors["Unknown"]
            
            # Draw player
            cv2.circle(court_img, (px, py), 15, color, -1)
            cv2.circle(court_img, (px, py), 15, (0, 0, 0), 2)
            
            # Draw player number
            cv2.putText(
                court_img,
                str(i + 1),
                (px - 8, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return court_img
