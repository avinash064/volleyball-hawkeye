"""
Re-Identification (Re-ID) Feature Extractor

Extracts appearance features from player crops for robust re-identification
after occlusions or ID switches.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict
import cv2
from pathlib import Path


class ReIDFeatureExtractor:
    """
    Extract appearance features for player re-identification.
    
    Uses a lightweight CNN to extract appearance embeddings that are
    invariant to pose changes but distinctive for different players.
    """
    
    def __init__(self, model_name: str = "resnet18", device: str = "cuda:0"):
        """
        Initialize Re-ID feature extractor.
        
        Args:
            model_name: Backbone model ('resnet18', 'mobilenet_v3_small')
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load pretrained model and modify for feature extraction
        if model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Remove classification layer
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        elif model_name == "mobilenet_v3_small":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Remove classifier
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 576
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard Re-ID size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✅ Re-ID extractor initialized ({model_name}) on {self.device}")
    
    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract Re-ID features from player crops.
        
        Args:
            crops: List of player crop images (BGR format)
        
        Returns:
            Feature matrix (N x feature_dim)
        """
        if len(crops) == 0:
            return np.array([])
        
        # Preprocess crops
        batch = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Transform
            tensor = self.transform(crop_rgb)
            batch.append(tensor)
        
        if len(batch) == 0:
            return np.array([])
        
        # Stack batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
            features = nn.functional.normalize(features, p=2, dim=1)  # L2 normalize
        
        return features.cpu().numpy()
    
    def compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between two feature sets.
        
        Args:
            features1: First feature matrix (N x D)
            features2: Second feature matrix (M x D)
        
        Returns:
            Similarity matrix (N x M), values in [0, 1]
        """
        # Cosine similarity = normalized dot product
        similarity = np.dot(features1, features2.T)
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity


class ReIDMatcher:
    """
    Match detections to tracks using Re-ID features.
    
    Combines IoU and appearance similarity for robust tracking.
    """
    
    def __init__(
        self,
        feature_extractor: ReIDFeatureExtractor,
        iou_weight: float = 0.4,
        reid_weight: float = 0.6,
        reid_threshold: float = 0.5
    ):
        """
        Initialize Re-ID matcher.
        
        Args:
            feature_extractor: Re-ID feature extractor
            iou_weight: Weight for IoU similarity (0-1)
            reid_weight: Weight for Re-ID similarity (0-1)
            reid_threshold: Minimum Re-ID similarity to consider match
        """
        self.feature_extractor = feature_extractor
        self.iou_weight = iou_weight
        self.reid_weight = reid_weight
        self.reid_threshold = reid_threshold
        
        # Normalize weights
        total = iou_weight + reid_weight
        self.iou_weight /= total
        self.reid_weight /= total
    
    def match(
        self,
        detections: List[Dict],
        tracks: List[Dict],
        iou_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute matching scores combining IoU and Re-ID.
        
        Args:
            detections: List of detections with 'crop' field
            tracks: List of tracks with 'features' field
            iou_matrix: Pre-computed IoU matrix (N_tracks x N_dets)
        
        Returns:
            Combined similarity matrix (N_tracks x N_dets)
        """
        n_tracks, n_dets = iou_matrix.shape
        
        if n_tracks == 0 or n_dets == 0:
            return iou_matrix
        
        # Extract features from detections
        det_crops = [d.get("crop") for d in detections]
        det_features = self.feature_extractor.extract_features(det_crops)
        
        if len(det_features) == 0:
            # Fall back to IoU only
            return iou_matrix
        
        # Get features from tracks
        track_features = []
        for track in tracks:
            feat = track.get("features")
            if feat is not None:
                track_features.append(feat)
            else:
                # No features, use zero vector
                track_features.append(np.zeros(self.feature_extractor.feature_dim))
        
        track_features = np.array(track_features)
        
        # Compute Re-ID similarity
        reid_matrix = self.feature_extractor.compute_similarity(
            track_features, det_features
        )
        
        # Apply threshold
        reid_matrix[reid_matrix < self.reid_threshold] = 0
        
        # Combine IoU and Re-ID
        combined = (
            self.iou_weight * iou_matrix +
            self.reid_weight * reid_matrix
        )
        
        return combined
