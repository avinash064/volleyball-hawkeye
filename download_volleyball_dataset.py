"""
Download volleyball tracking dataset from Roboflow

Uses user's specific dataset for fine-tuning.
"""

from roboflow import Roboflow

print("=" * 60)
print("Downloading Volleyball Tracking Dataset")
print("=" * 60)

# Initialize Roboflow with API key
rf = Roboflow(api_key="oQ2dhIWqNhYAGGh2bQwn")

# Access project
print("\n[1/3] Connecting to Roboflow workspace...")
project = rf.workspace("avinash064").project("tracking-volleyball-players-fk69q")
print("[OK] Connected to project: tracking-volleyball-players-fk69q")

# Get version 3
print("\n[2/3] Downloading dataset version 3...")
version = project.version(3)

# Download in YOLOv11 format (compatible with YOLOv8)
dataset = version.download("yolov11", location="datasets/volleyball_roboflow")

print("\n[3/3] Dataset downloaded!")
print("=" * 60)
print(f"\n[OK] Dataset location: {dataset.location}")
print(f"[OK] Dataset ready for training")

print("\n" + "=" * 60)
print("Next step: Start training!")
print("=" * 60)
print("\nRun:")
print("  python scripts/train_volleyball.py --data datasets/volleyball_roboflow/data.yaml --epochs 50")
