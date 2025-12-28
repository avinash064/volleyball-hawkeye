"""
Visualize VIA (VGG Image Annotator) annotations on images

Reads via_region_data.json and overlays annotations on images,
saving visualized results to output folder.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import os

def visualize_via_annotations(
    data_folder,
    annotation_file="via_region_data.json",
    output_folder="output"
):
    """
    Visualize VIA annotations on images.
    
    Args:
        data_folder: Folder containing images and annotation JSON
        annotation_file: Name of VIA annotation JSON file
        output_folder: Where to save visualized images
    """
    data_path = Path(data_folder)
    annotation_path = data_path / annotation_file
    output_path = Path(data_folder).parent / output_folder
    
    print("=" * 60)
    print("VIA Annotation Visualizer")
    print("=" * 60)
    print(f"\nData folder: {data_path}")
    print(f"Annotation file: {annotation_path}")
    print(f"Output folder: {output_path}")
    
    # Create output directory
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load annotations
    if not annotation_path.exists():
        print(f"\n[ERROR] Annotation file not found: {annotation_path}")
        return
    
    print(f"\n[1/3] Loading annotations...")
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"[OK] Loaded annotations for {len(annotations)} images")
    
    # Process each image
    print(f"\n[2/3] Processing images...")
    processed = 0
    
    for img_id, img_data in annotations.items():
        filename = img_data.get('filename')
        if not filename:
            continue
        
        img_path = data_path / filename
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {filename}")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] Failed to load: {filename}")
            continue
        
        # Draw annotations
        regions = img_data.get('regions', [])
        
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Get label
            label = region_attrs.get('label', region_attrs.get('class', 'object'))
            
            # Draw based on shape type
            shape_name = shape_attrs.get('name', '')
            
            if shape_name == 'rect':
                # Rectangle
                x = int(shape_attrs.get('x', 0))
                y = int(shape_attrs.get('y', 0))
                w = int(shape_attrs.get('width', 0))
                h = int(shape_attrs.get('height', 0))
                
                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif shape_name == 'polygon':
                # Polygon
                x_points = shape_attrs.get('all_points_x', [])
                y_points = shape_attrs.get('all_points_y', [])
                
                if len(x_points) == len(y_points) and len(x_points) > 0:
                    points = np.array([[int(x), int(y)] for x, y in zip(x_points, y_points)])
                    
                    # Draw polygon
                    cv2.polylines(image, [points], True, (0, 255, 0), 2)
                    
                    # Draw label at first point
                    if len(points) > 0:
                        cv2.putText(image, label, tuple(points[0]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif shape_name == 'circle':
                # Circle
                cx = int(shape_attrs.get('cx', 0))
                cy = int(shape_attrs.get('cy', 0))
                r = int(shape_attrs.get('r', 0))
                
                # Draw circle
                cv2.circle(image, (cx, cy), r, (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(image, label, (cx - r, cy - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif shape_name == 'ellipse':
                # Ellipse
                cx = int(shape_attrs.get('cx', 0))
                cy = int(shape_attrs.get('cy', 0))
                rx = int(shape_attrs.get('rx', 0))
                ry = int(shape_attrs.get('ry', 0))
                
                # Draw ellipse
                cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(image, label, (cx - rx, cy - ry - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif shape_name == 'point':
                # Point
                cx = int(shape_attrs.get('cx', 0))
                cy = int(shape_attrs.get('cy', 0))
                
                # Draw point
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                
                # Draw label
                cv2.putText(image, label, (cx + 10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add info overlay
        info_text = f"{filename} - {len(regions)} annotations"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Save visualized image
        output_file = output_path / f"annotated_{filename}"
        cv2.imwrite(str(output_file), image)
        
        processed += 1
        print(f"  [OK] {filename} -> {output_file.name} ({len(regions)} regions)")
    
    print(f"\n[3/3] Complete!")
    print("=" * 60)
    print(f"Processed: {processed} images")
    print(f"Output folder: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Configuration
    data_folder = r"C:\Users\xghostrider\Downloads\exercise1\exercise_1\617101a0d45ab471674827d3"
    
    # Run visualization
    visualize_via_annotations(data_folder)
    
    print("\n[OK] Visualization complete! Check the output folder.")
