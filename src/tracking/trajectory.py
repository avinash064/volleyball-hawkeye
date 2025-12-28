"""
Ball Trajectory Prediction and Visualization

Predicts volleyball trajectory using physics and Kalman filtering.
"""

import numpy as np
from typing import List, Tuple, Optional, Deque
from collections import deque
import cv2


class KalmanFilter:
    """Simple Kalman filter for ball tracking"""
    
    def __init__(self, dt: float = 1.0 / 30):
        """
        Initialize Kalman filter for ball trajectory.
        
        State: [x, y, vx, vy, ax, ay]
        
        Args:
            dt: Time step (default: 1/30s for 30fps)
        """
        self.dt = dt
        
        # State dimension: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe only position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(2) * 10.0
        
        # Error covariance
        self.P = np.eye(6) * 100
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # Update error covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P


class BallTrajectoryPredictor:
    """
    Predict volleyball trajectory using physics-based model.
    
    Accounts for gravity and air resistance.
    """
    
    def __init__(
        self,
        gravity: float = 9.81,  # m/s^2
        fps: int = 30,
        max_history: int = 60,
        use_kalman: bool = True
    ):
        """
        Initialize ball trajectory predictor.
        
        Args:
            gravity: Gravitational acceleration (m/s^2)
            fps: Video frame rate
            max_history: Maximum trajectory history to keep
            use_kalman: Use Kalman filter for smoothing
        """
        self.gravity = gravity
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_history = max_history
        
        # Trajectory history (pixel coordinates)
        self.history: Deque[Tuple[float, float]] = deque(maxlen=max_history)
        
        # Kalman filter for smoothing
        self.kalman = KalmanFilter(dt=self.dt) if use_kalman else None
        
        self.last_position = None
        self.last_velocity = None
    
    def update(self, position: Tuple[float, float]):
        """
        Update trajectory with new ball position.
        
        Args:
            position: Ball center (x, y) in pixels
        """
        if self.kalman:
            # Use Kalman filter
            self.kalman.update(np.array(position))
            filtered_pos = self.kalman.state[:2]
            self.history.append(tuple(filtered_pos))
            
            # Update velocity estimate
            self.last_velocity = self.kalman.state[2:4]
        else:
            # Simple history tracking
            self.history.append(position)
            
            # Estimate velocity from recent positions
            if len(self.history) >= 2:
                p1 = np.array(self.history[-2])
                p2 = np.array(self.history[-1])
                self.last_velocity = (p2 - p1) / self.dt
        
        self.last_position = position
    
    def predict(self, n_steps: int = 30) -> List[Tuple[float, float]]:
        """
        Predict future ball positions.
        
        Args:
            n_steps: Number of future frames to predict
        
        Returns:
            List of predicted (x, y) positions
        """
        if len(self.history) < 3 or self.last_velocity is None:
            # Not enough data to predict
            return []
        
        if self.kalman:
            # Use Kalman filter prediction
            predictions = []
            state = self.kalman.state.copy()
            F = self.kalman.F
            
            for _ in range(n_steps):
                state = F @ state
                predictions.append((state[0], state[1]))
            
            return predictions
        else:
            # Simple ballistic prediction
            predictions = []
            
            pos = np.array(self.last_position)
            vel = self.last_velocity.copy()
            
            # Gravity in pixel space (estimate)
            # Assumes ~50 pixels per meter (rough estimate)
            gravity_pixels = self.gravity * 50 * self.dt  # pixels/frame
            
            for _ in range(n_steps):
                # Update velocity (gravity only affects y)
                vel[1] += gravity_pixels
                
                # Update position
                pos += vel * self.dt
                
                predictions.append(tuple(pos))
            
            return predictions
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get historical trajectory"""
        return list(self.history)
    
    def reset(self):
        """Reset trajectory predictor"""
        self.history.clear()
        if self.kalman:
            self.kalman = KalmanFilter(dt=self.dt)
        self.last_position = None
        self.last_velocity = None


def draw_ball_trajectory(
    image: np.ndarray,
    trajectory: List[Tuple[float, float]],
    prediction: Optional[List[Tuple[float, float]]] = None,
    history_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
    prediction_color: Tuple[int, int, int] = (0, 165, 255),  # Orange
    thickness: int = 2
) -> np.ndarray:
    """
    Draw ball trajectory on image.
    
    Args:
        image: Input image
        trajectory: Historical trajectory points
        prediction: Future predicted points (optional)
        history_color: Color for historical trajectory
        prediction_color: Color for predicted trajectory
        thickness: Line thickness
    
    Returns:
        Image with trajectory drawn
    """
    overlay = image.copy()
    
    # Draw historical trajectory
    if len(trajectory) > 1:
        points = np.array(trajectory, dtype=np.int32)
        
        # Draw fading trail
        for i in range(len(points) - 1):
            alpha = (i + 1) / len(points)  # Fade from 0 to 1
            color = tuple(int(c * alpha) for c in history_color)
            cv2.line(overlay, tuple(points[i]), tuple(points[i + 1]), color, thickness)
        
        # Draw dots at positions
        for pt in points:
            cv2.circle(overlay, tuple(pt), 3, history_color, -1)
    
    # Draw predicted trajectory
    if prediction and len(prediction) > 0:
        pred_points = np.array(prediction, dtype=np.int32)
        
        # Draw dashed line for prediction
        for i in range(len(pred_points) - 1):
            if i % 2 == 0:  # Dashed effect
                cv2.line(
                    overlay,
                    tuple(pred_points[i]),
                    tuple(pred_points[i + 1]),
                    prediction_color,
                    thickness
                )
        
        # Draw landing point estimate
        if len(pred_points) > 0:
            cv2.circle(overlay, tuple(pred_points[-1]), 8, prediction_color, 2)
            cv2.circle(overlay, tuple(pred_points[-1]), 4, (255, 255, 255), -1)
    
    return overlay
