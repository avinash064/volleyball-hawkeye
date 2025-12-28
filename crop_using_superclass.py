"""
Crop Images Using Super-Class Annotations

Reads superclass_annotation.json from each subfolder and creates crops
organized by super-class labels.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def process_subfolder(subfolder_path, output_base):
    """
    Process a single subfolder: read superclass annotations and create crops.
    
    Args:
        subfolder_path: Path to subfolder
        output_base: Base output directory
    
    Returns:
        dict: Statistics per super-class
    """
    annotation_file = subfolder_path / "superclass_annotation.json"
    
    if not annotation_file.exists():
        return {}
    
    # Load super-class annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    image_name = data.get('image')
    annotations = data.get('annotations', [])
    
    if not image_name or not annotations:
        return {}
    
    # Load image
    img_path = subfolder_path / image_name
    if not img_path.exists():
        return {}
    
    image = cv2.imread(str(img_path))
    if image is None:
        return {}
    
    h, w = image.shape[:2]
    stats = defaultdict(int)
    
    # Process each super-class annotation
    for annotation in annotations:
        label = annotation.get('label')
        bbox = annotation.get('bbox')
        
        if not label or not bbox or len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = bbox
        
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
        
        # Create output directory for this super-class
        output_dir = output_base / label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        img_stem = Path(image_name).stem
        crop_filename = f"{subfolder_path.name}_{img_stem}_{label.replace(' ', '_')}.jpg"
        crop_path = output_dir / crop_filename
        
        # Save crop
        cv2.imwrite(str(crop_path), crop)
        stats[label] += 1
    
    return stats


def main():
    """
    Main processing function.
    """
    # Configuration
    super_folder = Path(r"C:\Users\xghostrider\Downloads\exercise1\exercise_1")
    output_folder = super_folder / "superclass_crops"
    
    print("=" * 70)
    print("SUPER-CLASS CROP EXTRACTOR")
    print("=" * 70)
    print(f"\nInput folder: {super_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output base directory
    output_folder.mkdir(exist_ok=True)
    
    # Find all subfolders
    subfolders = [d for d in super_folder.iterdir() 
                  if d.is_dir() and d.name not in ["output", "organized_crops", 
                                                     "view_based_crops", "superclass_crops"]]
    
    print(f"\nFound {len(subfolders)} subfolders to process")
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    total_stats = defaultdict(int)
    processed_count = 0
    
    # Process each subfolder
    for subfolder in subfolders:
        stats = process_subfolder(subfolder, output_folder)
        
        if stats:
            total = sum(stats.values())
            print(f"[OK] {subfolder.name} -> {total} crops")
            for label, count in sorted(stats.items()):
                total_stats[label] += count
            processed_count += 1
        else:
            print(f"[SKIP] {subfolder.name}")
    
    # Summary
    total_crops = sum(total_stats.values())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nProcessed: {processed_count} subfolders")
    print(f"Total crops: {total_crops}")
    print("\nBreakdown by super-class:")
    
    for label in sorted(total_stats.keys()):
        count = total_stats[label]
        percentage = (count / total_crops * 100) if total_crops > 0 else 0
        print(f"  {label:20s}: {count:5d} crops ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE:")
    print("=" * 70)
    for label in sorted(total_stats.keys()):
        folder_path = output_folder / label
        if folder_path.exists():
            num_files = len(list(folder_path.glob("*.jpg")))
            print(f"  {label}/  ({num_files} images)")
    
    print("\n" + "=" * 70)
    print(f"[OK] COMPLETE! All crops saved to:")
    print(f"     {output_folder}")
    print("=" * 70)
    print("\nFolder structure:")
    print(f"  {output_folder.name}/")
    for label in sorted(total_stats.keys()):
        print(f"    ├─ {label}/")
    print()


if __name__ == "__main__":
    main()
