"""
🧩 PUZZLE-BASED SUPER-CLASS CROPPING

Intelligent annotation merging with adjacency validation and overlap detection.
Treats each image as a puzzle with 1-2 active sections.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Super-class definitions
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

# Common classes (puzzle edge pieces)
COMMON_CLASSES = ["Roof", "roofrail", "alloywheel", "tyre", "wheelcap"]

# Adjacency rules (which super-classes can appear together)
ADJACENCY_RULES = {
    "Front": ["Front Left", "Front Right"],
    "Front Left": ["Front", "Rear Left"],
    "Front Right": ["Front", "Rear Right"],
    "Rear": ["Rear Left", "Rear Right"],
    "Rear Left": ["Front Left", "Rear"],
    "Rear Right": ["Front Right", "Rear"]
}

# Create reverse mapping
class_to_super = {}
for super_class, classes in SUPER_CLASSES.items():
    for cls in classes:
        class_to_super[cls] = super_class


def polygon_to_bbox(x_points, y_points):
    """Convert polygon to bounding box."""
    if not x_points or not y_points:
        return None
    x1, x2 = int(min(x_points)), int(max(x_points))
    y1, y2 = int(min(y_points)), int(max(y_points))
    return [x1, y1, x2, y2]


def merge_bboxes(bboxes):
    """Merge multiple bboxes into minimum enclosing rectangle."""
    if not bboxes:
        return None
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def compute_overlap_percentage(bbox1, bbox2):
    """
    Compute what percentage of bbox1 overlaps with bbox2.
    
    Returns:
        float: Percentage of bbox1 that overlaps with bbox2 (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No overlap
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if bbox1_area == 0:
        return 0.0
    
    return intersection_area / bbox1_area


def are_adjacent(super_class1, super_class2):
    """Check if two super-classes are adjacent."""
    if super_class1 == super_class2:
        return True
    allowed = ADJACENCY_RULES.get(super_class1, [])
    return super_class2 in allowed


def validate_active_trays(active_trays):
    """
    Validate that active super-classes follow rules:
    - Max 2 super-classes
    - Must be adjacent if 2
    
    Returns:
        list: Valid super-classes to keep
    """
    if len(active_trays) == 0:
        return []
    
    if len(active_trays) == 1:
        return active_trays
    
    if len(active_trays) == 2:
        if are_adjacent(active_trays[0], active_trays[1]):
            return active_trays
        else:
            # Non-adjacent - keep the one with more parts
            return [active_trays[0]]  # Fallback: keep first
    
    # More than 2 - keep first two if adjacent, otherwise just first
    if are_adjacent(active_trays[0], active_trays[1]):
        return active_trays[:2]
    else:
        return [active_trays[0]]


def process_subfolder(subfolder_path, output_base):
    """
    Process subfolder with puzzle-based logic.
    
    Args:
        subfolder_path: Path to subfolder
        output_base: Output directory
    
    Returns:
        dict: Statistics
    """
    annotation_file = subfolder_path / "via_region_data.json"
    
    if not annotation_file.exists():
        return {}
    
    # Load VIA annotations
    with open(annotation_file, 'r') as f:
        via_data = json.load(f)
    
    stats = defaultdict(int)
    
    # Process each image
    for img_id, img_data in via_data.items():
        filename = img_data.get('filename')
        if not filename:
            continue
        
        img_path = subfolder_path / filename
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        regions = img_data.get('regions', [])
        
        # STEP 1: Build super-class trays
        trays = defaultdict(list)  # super_class -> list of bboxes
        common_pieces = []  # common class bboxes
        
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            identity = region_attrs.get('identity')
            if not identity:
                continue
            
            # Extract bbox from polygon
            if shape_attrs.get('name') == 'polygon':
                x_points = shape_attrs.get('all_points_x', [])
                y_points = shape_attrs.get('all_points_y', [])
                bbox = polygon_to_bbox(x_points, y_points)
            else:
                # Handle other shapes (rect, circle, etc.)
                continue
            
            if bbox is None:
                continue
            
            # Categorize into trays
            if identity in COMMON_CLASSES:
                common_pieces.append(bbox)
            else:
                super_class = class_to_super.get(identity)
                if super_class:
                    trays[super_class].append(bbox)
        
        # STEP 2: Decide active trays
        active_trays = [sc for sc in trays.keys() if trays[sc]]
        
        # Sort by number of parts (descending) for consistency
        active_trays = sorted(active_trays, key=lambda sc: len(trays[sc]), reverse=True)
        
        # Validate and limit to 1-2 adjacent
        active_trays = validate_active_trays(active_trays)
        
        if not active_trays:
            continue
        
        # STEP 3: Assemble each active tray
        for super_class in active_trays:
            # Merge all parts in this tray
            super_class_roi = merge_bboxes(trays[super_class])
            
            if super_class_roi is None:
                continue
            
            # STEP 4: Attach common pieces if they fit
            pieces_to_merge = trays[super_class].copy()
            
            for common_bbox in common_pieces:
                overlap = compute_overlap_percentage(common_bbox, super_class_roi)
                if overlap >= 0.3:  # ≥30% overlap
                    pieces_to_merge.append(common_bbox)
            
            # Final ROI with common pieces
            final_roi = merge_bboxes(pieces_to_merge)
            
            if final_roi is None:
                continue
            
            x1, y1, x2, y2 = final_roi
            
            # Clip to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # STEP 5: Crop and save
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Create output directory
            output_dir = output_base / super_class
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save crop
            img_stem = Path(filename).stem
            crop_filename = f"{img_stem}_{super_class.replace(' ', '')}.jpg"
            crop_path = output_dir / crop_filename
            
            cv2.imwrite(str(crop_path), crop)
            stats[super_class] += 1
    
    return stats


def main():
    """Main processing function."""
    super_folder = Path(r"C:\Users\xghostrider\Downloads\exercise1\exercise_1")
    output_folder = super_folder / "puzzle_crops"
    
    print("=" * 70)
    print("PUZZLE-BASED SUPER-CLASS CROPPING")
    print("=" * 70)
    print(f"\nInput: {super_folder}")
    print(f"Output: {output_folder}")
    
    output_folder.mkdir(exist_ok=True)
    
    # Find subfolders
    subfolders = [d for d in super_folder.iterdir() 
                  if d.is_dir() and d.name not in ["output", "organized_crops", 
                                                     "view_based_crops", "superclass_crops", "puzzle_crops"]]
    
    print(f"\nProcessing {len(subfolders)} subfolders...")
    print("\n" + "=" * 70)
    
    total_stats = defaultdict(int)
    processed = 0
    
    for subfolder in subfolders:
        stats = process_subfolder(subfolder, output_folder)
        if stats:
            total = sum(stats.values())
            print(f"[OK] {subfolder.name} -> {total} crops")
            for sc, count in stats.items():
                total_stats[sc] += count
            processed += 1
        else:
            print(f"[SKIP] {subfolder.name}")
    
    # Summary
    total_crops = sum(total_stats.values())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nProcessed: {processed} subfolders")
    print(f"Total crops: {total_crops}")
    print("\nBreakdown:")
    for sc in sorted(total_stats.keys()):
        count = total_stats[sc]
        pct = (count / total_crops * 100) if total_crops > 0 else 0
        print(f"  {sc:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETE! Crops saved to: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
