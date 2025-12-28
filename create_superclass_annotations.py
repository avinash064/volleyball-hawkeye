"""
Super-Class Annotation Generator

Merges fine-grained car part annotations into 6 view-based super-classes
and generates a new annotation JSON file per subfolder.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define the 6 super classes mapping
SUPER_CLASSES = {
    "Front": [
        "antenna", "bonnet", "frontbumper", "frontbumpercladding", "frontbumpergrille",
        "frontws", "partial_frontws", "headlightwasher", "wiper",
        "leftheadlamp", "rightheadlamp", "leftfoglamp", "rightfoglamp",
        "licenseplate", "logo", "namebadge", "nameb_audi_a3",
        "lowerbumpergrille", "sensor"
    ],
    "Front Left": [
        "leftfrontdoor", "leftfrontdoorcladding", "leftfrontdoorglass", "leftfrontventglass",
        "leftfender", "leftwa", "leftapillar",
        "leftorvm", "leftrunningboard",
        "doorhandle", "footstep", "indicator"
    ],
    "Rear Left": [
        "leftreardoor", "leftreardoorcladding", "leftreardoorglass", "leftrearventglass",
        "leftbpillar", "leftcpillar", "leftdpillar",
        "leftqpanel", "leftquarterglass",
        "leftroofside", "leftbootlamp"
    ],
    "Rear": [
        "rearbumper", "rearbumpercladding", "rearws", "partial_rearws",
        "tailgate", "partial_tailgate",
        "lefttaillamp", "righttaillamp",
        "leftbootlamp", "rightbootlamp",
        "towbarcover", "Reflector"
    ],
    "Rear Right": [
        "rightreardoor", "rightreardoorcladding", "rightreardoorglass", "rightrearventglass",
        "rightbpillar", "rightcpillar", "rightdpillar",
        "rightqpanel", "rightquarterglass", "partial_rightqpanel",
        "rightroofside", "rightbootlamp"
    ],
    "Front Right": [
        "rightfrontdoor", "rightfrontdoorcladding", "rightfrontdoorglass",
        "rightfrontventglass", "partial_rightfrontdoor",
        "rightfender", "rightwa", "rightapillar",
        "rightorvm", "rightrunningboard"
    ]
}

# Common classes that should be merged into all detected super-classes
COMMON_CLASSES = [
    "Roof", "roofrail",
    "alloywheel", "tyre", "wheelcap",
    "fuelcap", "doorglass",
    "scratch", "bumperdent", "bumpertear",
    "N"
]

# Create reverse mapping (class -> super class)
class_to_super = {}
for super_class, classes in SUPER_CLASSES.items():
    for cls in classes:
        class_to_super[cls] = super_class


def extract_bbox_from_shape(shape_attrs):
    """
    Extract bounding box from VIA shape attributes.
    
    Returns:
        list: [x1, y1, x2, y2] or None
    """
    shape_name = shape_attrs.get('name', '')
    
    if shape_name == 'rect':
        x = int(shape_attrs.get('x', 0))
        y = int(shape_attrs.get('y', 0))
        w = int(shape_attrs.get('width', 0))
        h = int(shape_attrs.get('height', 0))
        return [x, y, x + w, y + h]
    
    elif shape_name == 'polygon':
        x_points = shape_attrs.get('all_points_x', [])
        y_points = shape_attrs.get('all_points_y', [])
        
        if len(x_points) > 0 and len(y_points) > 0:
            x1, x2 = int(min(x_points)), int(max(x_points))
            y1, y2 = int(min(y_points)), int(max(y_points))
            return [x1, y1, x2, y2]
    
    elif shape_name == 'circle':
        cx = int(shape_attrs.get('cx', 0))
        cy = int(shape_attrs.get('cy', 0))
        r = int(shape_attrs.get('r', 0))
        return [cx - r, cy - r, cx + r, cy + r]
    
    elif shape_name == 'ellipse':
        cx = int(shape_attrs.get('cx', 0))
        cy = int(shape_attrs.get('cy', 0))
        rx = int(shape_attrs.get('rx', 0))
        ry = int(shape_attrs.get('ry', 0))
        return [cx - rx, cy - ry, cx + rx, cy + ry]
    
    return None


def merge_bboxes(bboxes):
    """
    Merge multiple bounding boxes into one minimum enclosing box.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] lists
    
    Returns:
        list: [x1, y1, x2, y2] enclosing all boxes
    """
    if not bboxes:
        return None
    
    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)
    
    return [x1, y1, x2, y2]


def process_subfolder(subfolder_path):
    """
    Process a single subfolder: read VIA annotations, merge by super-class,
    and create superclass_annotation.json.
    
    Args:
        subfolder_path: Path to subfolder
    
    Returns:
        bool: True if successful, False otherwise
    """
    annotation_file = subfolder_path / "via_region_data.json"
    
    if not annotation_file.exists():
        return False
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        via_annotations = json.load(f)
    
    # Process each image in the annotation file
    for img_id, img_data in via_annotations.items():
        filename = img_data.get('filename')
        if not filename:
            continue
        
        regions = img_data.get('regions', [])
        if not regions:
            continue
        
        # Group annotations by super-class
        super_class_bboxes = defaultdict(list)
        
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Get class identity
            identity = region_attrs.get('identity')
            if not identity:
                continue
            
            # Extract bbox
            bbox = extract_bbox_from_shape(shape_attrs)
            if bbox is None:
                continue
            
            # Skip common classes (they cause cross-view contamination)
            if identity in COMMON_CLASSES:
                continue
            
            # Map to super-class
            super_class = class_to_super.get(identity)
            if super_class is None:
                continue  # Ignore unknown classes
            
            super_class_bboxes[super_class].append(bbox)
        
        # Skip if no super-class detected
        if not super_class_bboxes:
            continue
        
        # Create super-class annotations
        annotations = []
        for super_class in sorted(super_class_bboxes.keys()):
            bboxes = super_class_bboxes[super_class]
            
            # Merge all bboxes into one minimum enclosing box
            merged_bbox = merge_bboxes(bboxes)
            if merged_bbox is None:
                continue
            
            annotations.append({
                "label": super_class,
                "bbox": merged_bbox
            })
        
        # Create output JSON
        output_data = {
            "image": filename,
            "annotations": annotations
        }
        
        # Save to superclass_annotation.json in the same subfolder
        output_file = subfolder_path / "superclass_annotation.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return True  # Successfully processed
    
    return False


def main():
    """
    Main processing function.
    """
    # Configuration
    super_folder = Path(r"C:\Users\xghostrider\Downloads\exercise1\exercise_1")
    
    print("=" * 70)
    print("SUPER-CLASS ANNOTATION GENERATOR")
    print("=" * 70)
    print(f"\nProcessing folder: {super_folder}")
    
    # Find all subfolders
    subfolders = [d for d in super_folder.iterdir() 
                  if d.is_dir() and d.name not in ["output", "organized_crops", "view_based_crops"]]
    
    print(f"\nFound {len(subfolders)} subfolders to process")
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    success_count = 0
    skip_count = 0
    
    # Process each subfolder
    for subfolder in subfolders:
        success = process_subfolder(subfolder)
        
        if success:
            print(f"[OK] {subfolder.name} -> superclass_annotation.json created")
            success_count += 1
        else:
            print(f"[SKIP] {subfolder.name} - no annotations or file not found")
            skip_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nSuccessfully processed: {success_count} subfolders")
    print(f"Skipped: {skip_count} subfolders")
    print(f"\nTotal: {success_count + skip_count}")
    
    print("\n" + "=" * 70)
    print("ANNOTATION FORMAT:")
    print("=" * 70)
    print("""{
  "image": "<image_name>",
  "annotations": [
    {
      "label": "Front",
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ]
}""")
    
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETE! Each subfolder now contains superclass_annotation.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
