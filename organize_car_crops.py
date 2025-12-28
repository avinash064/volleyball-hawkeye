"""
Organize and crop car part images into 6 super class folders

Processes all subfolders in exercise1, reads VIA annotations,
crops car parts, and organizes them by view category.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define the 6 super classes mapping (from previous organization)
SUPER_CLASSES = {
    "1_Front": [
        "frontbumper", "frontbumpercladding", "frontbumpergrille", "lowerbumpergrille",
        "frontws", "partial_frontws", "bonnet", "logo", "namebadge", "nameb_audi_a3",
        "leftheadlamp", "rightheadlamp", "leftfoglamp", "rightfoglamp",
        "headlightwasher", "wiper", "sensor", "antenna", "licenseplate"
    ],
    "2_Front_Left": [
        "leftfrontdoor", "leftfrontdoorcladding", "leftfrontdoorglass", "leftfrontventglass",
        "leftfender", "leftwa", "leftapillar", "doorhandle", "leftorvm",
        "leftrunningboard", "footstep", "indicator"
    ],
    "3_Rear_Left": [
        "leftreardoor", "leftreardoorcladding", "leftreardoorglass", "leftrearventglass",
        "leftbpillar", "leftcpillar", "leftqpanel", "leftquarterglass",
        "leftroofside", "leftbootlamp", "leftdpillar"
    ],
    "4_Rear": [
        "rearbumper", "rearbumpercladding", "rearws", "partial_rearws",
        "tailgate", "partial_tailgate", "lefttaillamp", "righttaillamp",
        "leftbootlamp", "rightbootlamp", "towbarcover", "Reflector"
    ],
    "5_Rear_Right": [
        "rightreardoor", "rightreardoorcladding", "rightreardoorglass", "rightrearventglass",
        "rightbpillar", "rightcpillar", "rightqpanel", "rightquarterglass",
        "rightroofside", "rightbootlamp", "rightdpillar", "partial_rightqpanel"
    ],
    "6_Front_Right": [
        "rightfrontdoor", "rightfrontdoorcladding", "rightfrontdoorglass", "rightfrontventglass",
        "rightfender", "rightwa", "rightapillar", "rightorvm",
        "rightrunningboard", "partial_rightfrontdoor"
    ],
    "Common": [
        "Roof", "roofrail", "alloywheel", "tyre", "wheelcap", "fuelcap",
        "doorglass", "scratch", "bumperdent", "bumpertear", "N"
    ]
}

# Create reverse mapping (class -> super class)
class_to_super = {}
for super_class, classes in SUPER_CLASSES.items():
    for cls in classes:
        class_to_super[cls] = super_class


def process_subfolder(subfolder_path, output_base):
    """
    Process a single subfolder: read annotations, crop parts, organize by super class.
    
    Args:
        subfolder_path: Path to subfolder containing images and via_region_data.json
        output_base: Base output directory for organized crops
    """
    annotation_file = subfolder_path / "via_region_data.json"
    
    if not annotation_file.exists():
        print(f"  [SKIP] No annotation file: {subfolder_path.name}")
        return 0, defaultdict(int)
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    crops_saved = 0
    super_class_counts = defaultdict(int)
    
    # Process each image
    for img_id, img_data in annotations.items():
        filename = img_data.get('filename')
        if not filename:
            continue
        
        img_path = subfolder_path / filename
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        regions = img_data.get('regions', [])
        
        # Process each region/annotation
        for idx, region in enumerate(regions):
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # Get class identity
            identity = region_attrs.get('identity', 'unknown')
            if identity == 'unknown':
                continue
            
            # Map to super class
            super_class = class_to_super.get(identity, 'Common')
            
            # Extract bounding box based on shape type
            shape_name = shape_attrs.get('name', '')
            bbox = None
            
            if shape_name == 'rect':
                x = int(shape_attrs.get('x', 0))
                y = int(shape_attrs.get('y', 0))
                w = int(shape_attrs.get('width', 0))
                h = int(shape_attrs.get('height', 0))
                bbox = (x, y, x + w, y + h)
            
            elif shape_name == 'polygon':
                x_points = shape_attrs.get('all_points_x', [])
                y_points = shape_attrs.get('all_points_y', [])
                
                if len(x_points) > 0 and len(y_points) > 0:
                    x1, x2 = int(min(x_points)), int(max(x_points))
                    y1, y2 = int(min(y_points)), int(max(y_points))
                    bbox = (x1, y1, x2, y2)
            
            elif shape_name == 'circle':
                cx = int(shape_attrs.get('cx', 0))
                cy = int(shape_attrs.get('cy', 0))
                r = int(shape_attrs.get('r', 0))
                bbox = (cx - r, cy - r, cx + r, cy + r)
            
            elif shape_name == 'ellipse':
                cx = int(shape_attrs.get('cx', 0))
                cy = int(shape_attrs.get('cy', 0))
                rx = int(shape_attrs.get('rx', 0))
                ry = int(shape_attrs.get('ry', 0))
                bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
            
            if bbox is None:
                continue
            
            # Ensure bbox is within image bounds
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
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
            
            # Create output directory for this super class
            output_dir = output_base / super_class
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            crop_filename = f"{subfolder_path.name}_{Path(filename).stem}_{identity}_{idx}.jpg"
            crop_path = output_dir / crop_filename
            
            # Save crop
            cv2.imwrite(str(crop_path), crop)
            
            crops_saved += 1
            super_class_counts[super_class] += 1
    
    return crops_saved, super_class_counts


def main():
    """
    Process all subfolders in exercise1 and organize cropped parts.
    """
    # Configuration
    super_folder = Path(r"C:\Users\xghostrider\Downloads\exercise1\exercise_1")
    output_folder = super_folder / "organized_crops"
    
    print("=" * 70)
    print("CAR PART CROP ORGANIZER - 6 SUPER CLASSES")
    print("=" * 70)
    print(f"\nInput folder: {super_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output base directory
    output_folder.mkdir(exist_ok=True)
    
    # Find all subfolders
    subfolders = [d for d in super_folder.iterdir() 
                  if d.is_dir() and d.name != "output" and d.name != "organized_crops"]
    
    print(f"\nFound {len(subfolders)} subfolders to process")
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    total_crops = 0
    total_super_class_counts = defaultdict(int)
    
    # Process each subfolder
    for subfolder in subfolders:
        print(f"\n[Folder] {subfolder.name}")
        crops, counts = process_subfolder(subfolder, output_folder)
        
        if crops > 0:
            print(f"  [OK] Extracted {crops} crops")
            for super_class, count in sorted(counts.items()):
                print(f"     - {super_class}: {count} crops")
                total_super_class_counts[super_class] += count
            total_crops += crops
        else:
            print(f"  [WARN] No crops extracted")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal crops extracted: {total_crops}")
    print("\nBreakdown by super class:")
    for super_class in sorted(total_super_class_counts.keys()):
        count = total_super_class_counts[super_class]
        percentage = (count / total_crops * 100) if total_crops > 0 else 0
        print(f"  {super_class:20s}: {count:5d} crops ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE:")
    print("=" * 70)
    for super_class in sorted(SUPER_CLASSES.keys()):
        folder_path = output_folder / super_class
        if folder_path.exists():
            num_files = len(list(folder_path.glob("*.jpg")))
            print(f"  {output_folder.name}/{super_class}/  ({num_files} images)")
    
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETE! All crops saved to: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
