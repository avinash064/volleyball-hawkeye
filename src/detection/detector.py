"""
YOLO Detector Wrapper for Volleyball Detection

This module provides a clean interface for YOLOv8 detection with
volleyball-specific configurations and fallback strategies.
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class VolleyballDetector:
    """
    YOLOv8-based detector for volleyball players, ball, and referees.
    
    Supports both fine-tuned and COCO-pretrained models with automatic
    class mapping and confidence threshold adjustments per class.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_player: float = 0.35,
        conf_ball: float = 0.15,
        conf_referee: float = 0.30,
        imgsz: int = 960,
        device: str = "cuda:0"
    ):
        """
        Initialize volleyball detector.
        
        Args:
            model_path: Path to YOLO weights (fine-tuned or COCO)
            conf_player: Confidence threshold for players
            conf_ball: Confidence threshold for ball (lower for recall)
            conf_referee: Confidence threshold for referees
            imgsz: Input image size for inference
            device: Device to run inference on ('cuda:0', 'cpu')
        """
        self.model = YOLO(model_path)
        self.conf_player = conf_player
        self.conf_ball = conf_ball
        self.conf_referee = conf_referee
        self.imgsz = imgsz
        self.device = device
        
        # Determine if using COCO or fine-tuned model
        self.class_names = self.model.names
        self.is_coco = "person" in self.class_names.values()
        
        # Map COCO classes to volleyball classes if needed
        if self.is_coco:
            self.class_mapping = {
                0: "player",      # person -> player/referee
                32: "ball"        # sports ball -> ball
            }
            print("ℹ️ Using COCO-pretrained model (zero-training mode)")
        else:
            self.class_mapping = {
                0: "player",
                1: "ball",
                2: "referee"
            }
            print("ℹ️ Using fine-tuned volleyball model")
        
        print(f"✅ Detector initialized on {device}")
        print(f"   Player conf: {conf_player}")
        print(f"   Ball conf: {conf_ball}")
        print(f"   Referee conf: {conf_referee}")
    
    def detect(
        self,
        frame: np.ndarray,
        return_crops: bool = False
    ) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input image (BGR format)
            return_crops: If True, return cropped images for each detection
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - conf: confidence score
                - class: 'player', 'ball', or 'referee'
                - crop: cropped image (if return_crops=True)
        """
        # Run inference
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=min(self.conf_ball, self.conf_player, self.conf_referee),
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            # Map class
            if self.is_coco:
                if cls_id == 0:  # person
                    class_name = "player"  # Will classify as player/referee later
                    threshold = self.conf_player
                elif cls_id == 32:  # sports ball
                    class_name = "ball"
                    threshold = self.conf_ball
                else:
                    continue  # Skip other COCO classes
            else:
                class_name = self.class_mapping.get(cls_id, "unknown")
                if class_name == "player":
                    threshold = self.conf_player
                elif class_name == "ball":
                    threshold = self.conf_ball
                elif class_name == "referee":
                    threshold = self.conf_referee
                else:
                    continue
            
            # Apply class-specific threshold
            if conf < threshold:
                continue
            
            detection = {
                "bbox": bbox.tolist(),
                "conf": conf,
                "class": class_name
            }
            
            # Add crop if requested
            if return_crops:
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                detection["crop"] = crop
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """
        Detect objects in multiple frames (batched inference).
        
        Args:
            frames: List of input images
            batch_size: Batch size for inference
        
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            # Run batched inference
            results = self.model.predict(
                batch,
                imgsz=self.imgsz,
                conf=min(self.conf_ball, self.conf_player, self.conf_referee),
                device=self.device,
                verbose=False
            )
            
            # Process each result
            for result in results:
                detections = self._process_result(result)
                all_detections.append(detections)
        
        return all_detections
    
    def _process_result(self, result) -> List[Dict]:
        """Process a single YOLO result"""
        detections = []
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            # Map class and apply threshold (same logic as detect())
            if self.is_coco:
                if cls_id == 0:
                    class_name = "player"
                    threshold = self.conf_player
                elif cls_id == 32:
                    class_name = "ball"
                    threshold = self.conf_ball
                else:
                    continue
            else:
                class_name = self.class_mapping.get(cls_id, "unknown")
                if class_name == "player":
                    threshold = self.conf_player
                elif class_name == "ball":
                    threshold = self.conf_ball
                elif class_name == "referee":
                    threshold = self.conf_referee
                else:
                    continue
            
            if conf < threshold:
                continue
            
            detections.append({
                "bbox": bbox.tolist(),
                "conf": conf,
                "class": class_name
            })
        
        return detections
    
    def get_foot_points(self, detections: List[Dict]) -> np.ndarray:
        """
        Extract foot points from player detections for court projection.
        
        Args:
            detections: List of player detections
        
        Returns:
            Array of foot points (bottom-center of bboxes)
        """
        foot_points = []
        
        for det in detections:
            if det["class"] in ["player", "referee"]:
                x1, y1, x2, y2 = det["bbox"]
                foot_x = (x1 + x2) / 2
                foot_y = y2  # Bottom of bbox
                foot_points.append([foot_x, foot_y])
        
        return np.array(foot_points)
