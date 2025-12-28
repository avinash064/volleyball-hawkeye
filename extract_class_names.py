"""
Extract unique class names from VIA annotation file
"""

import json
from pathlib import Path
from collections import Counter

# Load annotation file
annotation_path = r"C:\Users\xghostrider\Downloads\exercise1\exercise_1\617101a0d45ab471674827d3\via_region_data.json"

print("=" * 60)
print("VIA Annotation Class Extractor")
print("=" * 60)

with open(annotation_path, 'r') as f:
    annotations = json.load(f)

# Collect all class names
class_names = []
class_counts = Counter()

for img_id, img_data in annotations.items():
    regions = img_data.get('regions', [])
    
    for region in regions:
        region_attrs = region.get('region_attributes', {})
        
        # Extract identity field (VIA format)
        label = region_attrs.get('identity', 'unlabeled')
        
        class_names.append(label)
        class_counts[label] += 1

# Get unique classes
unique_classes = sorted(set(class_names))

print(f"\nTotal annotations: {len(class_names)}")
print(f"Unique classes: {len(unique_classes)}")
print("\n" + "=" * 60)
print("CLASS NAMES:")
print("=" * 60)

for i, class_name in enumerate(unique_classes, 1):
    count = class_counts[class_name]
    percentage = (count / len(class_names)) * 100
    print(f"{i:2d}. {class_name:30s} - {count:4d} instances ({percentage:5.1f}%)")

print("=" * 60)

# Save to file
output_file = Path(annotation_path).parent.parent / "output" / "class_names.txt"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    f.write("VIA Annotation Classes\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total annotations: {len(class_names)}\n")
    f.write(f"Unique classes: {len(unique_classes)}\n\n")
    f.write("Class Names (sorted alphabetically):\n")
    f.write("-" * 60 + "\n")
    for class_name in unique_classes:
        f.write(f"- {class_name}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Class Distribution:\n")
    f.write("-" * 60 + "\n")
    for class_name in unique_classes:
        count = class_counts[class_name]
        percentage = (count / len(class_names)) * 100
        f.write(f"{class_name:30s} : {count:4d} ({percentage:5.1f}%)\n")

print(f"\n[OK] Class list saved to: {output_file}")
