"""
Organize car part classes into 6 super classes based on car views
"""

import json
from collections import defaultdict

# Define the 6 super classes based on car views
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
    
    "Common_All_Views": [
        "Roof", "roofrail", "alloywheel", "tyre", "wheelcap", "fuelcap",
        "doorglass", "scratch", "bumperdent", "bumpertear", "N"
    ]
}

# Create reverse mapping (class -> super class)
class_to_super = {}
for super_class, classes in SUPER_CLASSES.items():
    for cls in classes:
        class_to_super[cls] = super_class

print("=" * 70)
print("CAR PARTS ORGANIZED INTO 6 SUPER CLASSES")
print("=" * 70)

# Display organization
for super_class, classes in SUPER_CLASSES.items():
    print(f"\n{super_class} ({len(classes)} classes):")
    print("-" * 70)
    for cls in sorted(classes):
        print(f"  - {cls}")

# Count statistics
print("\n" + "=" * 70)
print("STATISTICS:")
print("=" * 70)
for super_class, classes in SUPER_CLASSES.items():
    print(f"{super_class:25s}: {len(classes):2d} classes")

total = sum(len(classes) for classes in SUPER_CLASSES.values())
print(f"\n{'Total':25s}: {total} classes")

# Save mapping to JSON
output_file = r"C:\Users\xghostrider\Downloads\exercise1\exercise_1\output\class_hierarchy.json"

hierarchy = {
    "super_classes": SUPER_CLASSES,
    "class_to_super_mapping": class_to_super,
    "statistics": {
        super_class: len(classes) 
        for super_class, classes in SUPER_CLASSES.items()
    }
}

with open(output_file, 'w') as f:
    json.dump(hierarchy, f, indent=2)

print(f"\n[OK] Hierarchy saved to: {output_file}")

# Create visual summary
summary_file = r"C:\Users\xghostrider\Downloads\exercise1\exercise_1\output\class_hierarchy.txt"

with open(summary_file, 'w') as f:
    f.write("CAR PARTS - 6 SUPER CLASSES HIERARCHY\n")
    f.write("=" * 70 + "\n\n")
    
    for super_class, classes in SUPER_CLASSES.items():
        f.write(f"{super_class} ({len(classes)} classes)\n")
        f.write("-" * 70 + "\n")
        for cls in sorted(classes):
            f.write(f"  - {cls}\n")
        f.write("\n")
    
    f.write("=" * 70 + "\n")
    f.write("STATISTICS\n")
    f.write("=" * 70 + "\n")
    for super_class, classes in SUPER_CLASSES.items():
        f.write(f"{super_class:25s}: {len(classes):2d} classes\n")
    f.write(f"\nTotal: {total} classes\n")

print(f"[OK] Summary saved to: {summary_file}")
