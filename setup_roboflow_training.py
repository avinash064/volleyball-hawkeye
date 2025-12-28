"""
Automated Roboflow Dataset Download and Training Setup

Downloads volleyball dataset from Roboflow and prepares for training.
Run this first before training.
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("Roboflow Dataset Download for Volleyball Training")
print("=" * 60)

# Check for API key
api_key = os.environ.get("ROBOFLOW_API_KEY")

if not api_key:
    print("\n🔑 Roboflow API Key Required")
    print("=" * 60)
    print("\nTo download the dataset, you need a free Roboflow API key.")
    print("\nSteps to get your API key:")
    print("  1. Go to: https://app.roboflow.com/")
    print("  2. Sign up or log in (free account)")
    print("  3. Click your profile → Settings → API")
    print("  4. Copy your API key")
    print("\nThen set it:")
    print("  Option 1 (PowerShell):")
    print("    $env:ROBOFLOW_API_KEY='your_key_here'")
    print("    python setup_roboflow_training.py")
    print("\n  Option 2 (Permanent):")
    print("    Add to system environment variables")
    print("\n  Option 3 (One-time):")
    print("    python datasets/roboflow/download_datasets.py --api-key your_key")
    print("\n" + "=" * 60)
    sys.exit(1)

print(f"\n✅ API key found: {api_key[:8]}...")
print("\n[1/3] Downloading volleyball dataset from Roboflow...")

# Download using the script
os.chdir("datasets/roboflow")
exit_code = os.system(f'python download_datasets.py --api-key "{api_key}"')

if exit_code != 0:
    print("\n❌ Download failed. Please check:")
    print("  - API key is correct")
    print("  - Internet connection")
    print("  - Roboflow service is available")
    sys.exit(1)

os.chdir("../..")

print("\n[2/3] Verifying dataset structure...")

# Check if dataset exists
dataset_dir = Path("datasets/volleyball")
if not dataset_dir.exists():
    print(f"❌ Dataset directory not found: {dataset_dir}")
    sys.exit(1)

# Check for required files
data_yaml = dataset_dir / "data.yaml"
train_imgs = dataset_dir / "images" / "train"
train_lbls = dataset_dir / "labels" / "train"

if not data_yaml.exists():
    print("⚠️ data.yaml not found, creating default...")
    # Will be created by download script or use existing

# Count images
if train_imgs.exists():
    num_train = len(list(train_imgs.glob("*.jpg"))) + len(list(train_imgs.glob("*.png")))
    print(f"✅ Training images: {num_train}")
else:
    print("❌ Training images directory not found")
    sys.exit(1)

if train_lbls.exists():
    num_labels = len(list(train_lbls.glob("*.txt")))
    print(f"✅ Training labels: {num_labels}")
else:
    print("❌ Training labels directory not found")
    sys.exit(1)

print("\n[3/3] Ready to train!")
print("=" * 60)
print("\n🚀 Start training with:")
print("  python scripts/train_volleyball.py")
print("\nOr customize:")
print("  python scripts/train_volleyball.py --epochs 50 --batch 16 --model-size l")
print("\n" + "=" * 60)
