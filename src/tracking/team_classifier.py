"""
Team Classification using Jersey Colors

Automatically classifies players into teams using K-means clustering
on jersey colors extracted from player crops.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import List, Dict
from collections import Counter


class TeamClassifier:
    """
    Classify players into teams based on jersey colors.
    
    Uses K-means clustering in HSV color space to separate players
    into Team A, Team B, and Referee.
    """
    
    def __init__(self, n_clusters: int = 3, color_space: str = "HSV"):
        """
        Initialize team classifier.
        
        Args:
            n_clusters: Number of teams (typically 3: Team A, B, Referee)
            color_space: Color space for clustering ('HSV', 'RGB', 'LAB')
        """
        self.n_clusters = n_clusters
        self.color_space = color_space
        self.kmeans = None
        self.cluster_labels = {}  # Map cluster ID to team name
        self.is_fitted = False
    
    def fit(self, player_crops: List[np.ndarray]):
        """
        Fit classifier on a batch of player crops.
        
        Args:
            player_crops: List of cropped player images
        """
        if len(player_crops) < self.n_clusters:
            print(f"⚠️ Not enough players ({len(player_crops)}) for {self.n_clusters} clusters")
            return
        
        # Extract dominant colors
        colors = []
        for crop in player_crops:
            color = self._extract_dominant_color(crop)
            if color is not None:
                colors.append(color)
        
        if len(colors) < self.n_clusters:
            print(f"⚠️ Could not extract enough colors")
            return
        
        colors = np.array(colors)
        
        # Cluster
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(colors)
        
        # Automatically label clusters (heuristic: largest cluster = Team A)
        labels = self.kmeans.labels_
        label_counts = Counter(labels)
        most_common = label_counts.most_common(self.n_clusters)
        
        # Assign labels
        self.cluster_labels = {
            most_common[0][0]: "Team A",
            most_common[1][0]: "Team B" if len(most_common) > 1 else "Unknown",
            most_common[2][0]: "Referee" if len(most_common) > 2 else "Unknown"
        }
        
        self.is_fitted = True
        print(f"✅ Team classifier fitted with {len(colors)} samples")
    
    def predict(self, player_crop: np.ndarray) -> str:
        """
        Predict team for a single player.
        
        Args:
            player_crop: Cropped player image
        
        Returns:
            Team name ('Team A', 'Team B', 'Referee')
        """
        if not self.is_fitted:
            return "Unknown"
        
        color = self._extract_dominant_color(player_crop)
        if color is None:
            return "Unknown"
        
        cluster_id = self.kmeans.predict([color])[0]
        return self.cluster_labels.get(cluster_id, "Unknown")
    
    def predict_batch(self, player_crops: List[np.ndarray]) -> List[str]:
        """Predict teams for multiple players"""
        return [self.predict(crop) for crop in player_crops]
    
    def _extract_dominant_color(self, image: np.ndarray) -> np.ndarray:
        """
        Extract dominant jersey color from player crop.
        
        Strategy:
        1. Focus on torso region (middle 50% vertically)
        2. Mask out skin tones
        3. Cluster colors and pick dominant
        
        Args:
            image: Player crop (BGR)
        
        Returns:
            Dominant color in specified color space
        """
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        
        # Focus on torso (skip head and legs)
        torso = image[int(h * 0.2):int(h * 0.6), :]
        
        if torso.size == 0:
            return None
        
        # Convert to target color space
        if self.color_space == "HSV":
            torso_converted = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        elif self.color_space == "LAB":
            torso_converted = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        else:  # RGB
            torso_converted = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
        
        # Remove skin tones (heuristic in HSV)
        if self.color_space == "HSV":
            # Skin tone mask
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 150, 255])
            mask = cv2.inRange(torso_converted, lower_skin, upper_skin)
            mask = cv2.bitwise_not(mask)
        else:
            mask = np.ones(torso_converted.shape[:2], dtype=np.uint8) * 255
        
        # Extract pixels
        pixels = torso_converted[mask > 0]
        
        if len(pixels) < 10:
            # Fallback: use entire crop
            pixels = torso_converted.reshape(-1, 3)
        
        # Cluster to find dominant color
        try:
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
        except:
            # Fallback: median color
            dominant_color = np.median(pixels, axis=0)
        
        return dominant_color
    
    def get_team_colors(self) -> Dict[str, np.ndarray]:
        """Get the representative color for each team"""
        if not self.is_fitted:
            return {}
        
        colors = {}
        for cluster_id, team_name in self.cluster_labels.items():
            color = self.kmeans.cluster_centers_[cluster_id]
            
            # Convert back to BGR for visualization
            if self.color_space == "HSV":
                color_bgr = cv2.cvtColor(
                    np.uint8([[color]]), cv2.COLOR_HSV2BGR
                )[0][0]
            elif self.color_space == "LAB":
                color_bgr = cv2.cvtColor(
                    np.uint8([[color]]), cv2.COLOR_LAB2BGR
                )[0][0]
            else:
                color_bgr = cv2.cvtColor(
                    np.uint8([[color]]), cv2.COLOR_RGB2BGR
                )[0][0]
            
            colors[team_name] = color_bgr.tolist()
        
        return colors
