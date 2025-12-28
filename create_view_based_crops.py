"""
Dataset Restructuring: Merge Annotations & Create View-Based Crops

For each image:
1. Merge all annotations of the same super-class
2. Include common classes in the relevant super-class
3. Create ONE enclosing crop per super-class
4. Save to view-specific folders
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

# Common classes that should be merged into relevant super-class
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
        tuple: (x1, y1, x2, y2) or None
    """
    shape_name = shape_attrs.get('name', '')
    
    if shape_name == 'rect':
        x = int(shape_attrs.get('x', 0))
        y = int(shape_attrs.get('y', 0))
        w = int(shape_attrs.get('width', 0))
        h = int(shape_attrs.get('height', 0))
        return (x, y, x + w, y + h)
    
    elif shape_name == 'polygon':
        x_points = shape_attrs.get('all_points_x', [])
        y_points = shape_attrs.get('all_points_y', [])
        
        if len(x_points) > 0 and len(y_points) > 0:
            x1, x2 = int(min(x_points)), int(max(x_points))
            y1, y2 = int(min(y_points)), int(max(y_points))
            return (x1, y1, x2, y2)
    
    elif shape_name == 'circle':
        cx = int(shape_attrs.get('cx', 0))
        cy = int(shape_attrs.get('cy', 0))
        r = int(shape_attrs.get('r', 0))
        return (cx - r, cy - r, cx + r, cy + r)
    
    elif shape_name == 'ellipse':
        cx = int(shape_attrs.get('cx', 0))
        cy = int(shape_attrs.get('cy', 0))
        rx = int(shape_attrs.get('rx', 0))
        ry = int(shape_attrs.get('ry', 0))
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    
    return None


def merge_bboxes(bboxes):
    """
    Merge multiple bounding boxes into one enclosing box.
    
    Args:
        bboxes: List of (x1, y1, x2, y2) tuples
    
    Returns:
        tuple: (x1, y1, x2, y2) enclosing all boxes
    """
    if not bboxes:
        return None
    
    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)
    
    return (x1, y1, x2, y2)


def process_image(img_path, annotations, output_base):
    """
    Process a single image: merge annotations by super-class and create crops.
    
    Args:
        img_path: Path to image file
        annotations: List of annotation regions
        output_base: Base output directory
    
    Returns:
        dict: Number of crops created per super-class
    """
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        return {}
    
    h, w = image.shape[:2]
    
    # Group annotations by super-class
    super_class_bboxes = defaultdict(list)
    
    for region in annotations:
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
        
        # Map to super-class
        super_class = class_to_super.get(identity)
        
        # If it's a common class, try to infer super-class from image context
        if super_class is None and identity in COMMON_CLASSES:
            # For now, assign to all present super-classes
            # This is a simplification - could be smarter based on bbox position
            for sc in SUPER_CLASSES.keys():
                if super_class_bboxes[sc]:  # If this super-class has other parts
                    super_class_bboxes[sc].append(bbox)
            # If no super-class detected yet, skip common parts
            continue
        
        if super_class is None:
            continue
        
        super_class_bboxes[super_class].append(bbox)
    
    # Create crops for each super-class
    crops_created = {}
    img_name = img_path.stem
    
    for super_class, bboxes in super_class_bboxes.items():
        if not bboxes:
            continue
        
        # Merge all bboxes into one enclosing box
        merged_bbox = merge_bboxes(bboxes)
        if merged_bbox is None:
            continue
        
        x1, y1, x2, y2 = merged_bbox
        
        # Ensure bbox is within image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop region
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Create output directory
        output_dir = output_base / super_class
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save crop
        crop_filename = f"{img_name}_{super_class.replace(' ', '_')}.jpg"
        crop_path = output_dir / crop_filename
        
        cv2.imwrite(str(crop_path), crop)
        crops_created[super_class] = 1
    
    return crops_created


def process_subfolder(subfolder_path, output_base):
    """
    Process a single subfolder.
    
    Args:
        subfolder_path: Path to subfolder
        output_base: Base output directory
    
    Returns:
        dict: Statistics per super-class
    """
    annotation_file = subfolder_path / "via_region_data.json"
    
    if not annotation_file.exists():
        return {}
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    stats = defaultdict(int)
    
    # Process each image
    for img_id, img_data in annotations.items():
        filename = img_data.get('filename')
        if not filename:
            continue
        
        img_path = subfolder_path / filename
        if not img_path.exists():
            continue
        
        regions = img_data.get('regions', [])
        if not regions:
            continue
        
        # Process this image
        crops_created = process_image(img_path, regions, output_base)
        
        # Update statistics
        for super_class, count in crops_created.items():
            stats[super_class] += count
    
    return stats


def main():
    """
    Main processing function.
    """
    # Configuration
    super_folder = Path(r"C:\Users\xghostrider\Downloads\exercise1\exercise_1")
    output_folder = super_folder / "view_based_crops"
    
    print("=" * 70)
    print("VIEW-BASED CROP ORGANIZER - MERGED ANNOTATIONS")
    print("=" * 70)
    print(f"\nInput folder: {super_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output base directory
    output_folder.mkdir(exist_ok=True)
    
    # Find all subfolders
    subfolders = [d for d in super_folder.iterdir() 
                  if d.is_dir() and d.name not in ["output", "organized_crops", "view_based_crops"]]
    
    print(f"\nFound {len(subfolders)} subfolders to process")
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    total_stats = defaultdict(int)
    
    # Process each subfolder
    for subfolder in subfolders:
        print(f"\n[Folder] {subfolder.name}")
        stats = process_subfolder(subfolder, output_folder)
        
        if stats:
            total = sum(stats.values())
            print(f"  [OK] Created {total} view-based crops")
            for super_class, count in sorted(stats.items()):
                print(f"     - {super_class}: {count} crops")
                total_stats[super_class] += count
        else:
            print(f"  [SKIP] No crops created")
    
    # Summary
    total_crops = sum(total_stats.values())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal view-based crops: {total_crops}")
    print("\nBreakdown by super-class:")
    for super_class in sorted(SUPER_CLASSES.keys()):
        count = total_stats.get(super_class, 0)
        percentage = (count / total_crops * 100) if total_crops > 0 else 0
        print(f"  {super_class:20s}: {count:5d} crops ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE:")
    print("=" * 70)
    for super_class in sorted(SUPER_CLASSES.keys()):
        folder_path = output_folder / super_class
        if folder_path.exists():
            num_files = len(list(folder_path.glob("*.jpg")))
            print(f"  {super_class}/  ({num_files} images)")
    
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETE! All view-based crops saved to:")
    print(f"     {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
