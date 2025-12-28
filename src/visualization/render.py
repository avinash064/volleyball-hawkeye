"""
Visualization utilities for rendering detections, tracks, and court views
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw a single bounding box with label"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
    
    # Draw label text
    cv2.putText(
        image, label, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    return image


def draw_trajectory(
    image: np.ndarray,
    trajectory: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """Draw trajectory trail"""
    if len(trajectory) < 2:
        return image
    
    points = np.array(trajectory, dtype=np.int32)
    
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color, thickness)
    
    return image


def create_side_by_side(
    frame: np.ndarray,
    court_view: np.ndarray,
    gap: int = 20
) -> np.ndarray:
    """Create side-by-side view of broadcast and court"""
    h1, w1 = frame.shape[:2]
    h2, w2 = court_view.shape[:2]
    
    # Resize court to match frame height
    court_resized = cv2.resize(court_view, (int(w2 * h1 / h2), h1))
    
    # Create combined image
    combined = np.ones((h1, w1 + court_resized.shape[1] + gap, 3), dtype=np.uint8) * 255
    
    # Place images
    combined[:, :w1] = frame
    combined[:, w1 + gap:] = court_resized
    
    # Draw divider
    cv2.line(combined, (w1 + gap // 2, 0), (w1 + gap // 2, h1), (200, 200, 200), gap)
    
    return combined


def add_info_panel(
    image: np.ndarray,
    info: Dict[str, str],
    position: str = "top-left"
) -> np.ndarray:
    """Add information panel to image"""
    panel_height = 30 + len(info) * 25
    panel_width = 300
    
    overlay = image.copy()
    
    if position == "top-left":
        x, y = 10, 10
    elif position == "top-right":
        x, y = image.shape[1] - panel_width - 10, 10
    else:
        x, y = 10, 10
    
    # Draw panel background
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_width, y + panel_height),
        (0, 0, 0),
        -1
    )
    
    # Blend
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw text
    line_y = y + 25
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(
            image, text, (x + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        line_y += 25
    
    return image
