"""
Demo: Complete Volleyball Training Workflow

This script demonstrates the entire volleyball training pipeline
from data preparation to model training.
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_workflow():
    """Demonstrate complete training workflow"""
    
    print_section("Volleyball Detection Training Demo")
    
    print("This demo shows three ways to train a volleyball detection model:\n")
    print("1. Download Roboflow datasets (fastest)")
    print("2. Extract and annotate your own videos (custom)")
    print("3. Quick test with COCO subset (testing)\n")
    
    # Check environment
    print_section("Step 1: Environment Check")
    
    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists():
        print("[OK] Virtual environment found")
    else:
        print("[ERROR] Virtual environment not found")
        print("   Run: python -m venv venv")
        return
    
    # Check if in volleyball directory
    if not Path("scripts").exists():
        print("[WARNING] Please run from Volleyball/ directory")
        return
    
    print("[OK] All checks passed!\n")
    
    # Show workflow options
    print_section("Training Workflow Options")
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OPTION 1: Roboflow Datasets (Recommended)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("Time: 15 min setup + 1-2 hours training")
    print("Quality: High (professional annotations)")
    print()
    print("Commands:")
    print("  1. Get API key: https://app.roboflow.com/settings/api")
    print("  2. Download:")
    print("     cd datasets/roboflow")
    print("     python download_datasets.py --api-key YOUR_KEY")
    print("  3. Train:")
    print("     cd ../..")
    print("     python scripts/train_volleyball.py")
    print()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OPTION 2: Custom Video Annotation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("Time: 3-5 hours annotation + 1-2 hours training")
    print("Quality: Custom to your footage")
    print()
    print("Commands:")
    print("  1. Extract frames:")
    print("     python scripts/extract_frames.py --video your_video.mp4")
    print("  2. Annotate:")
    print("     - CVAT: https://app.cvat.ai")
    print("     - LabelImg: labelImg")
    print("     - See ANNOTATION_GUIDE.md")
    print("  3. Export labels to datasets/volleyball/labels/train/")
    print("  4. Train:")
    print("     python scripts/train_volleyball.py")
    print()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OPTION 3: Quick Test (No Real Data)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("Time: 10-15 minutes")
    print("Quality: Testing only")
    print()
    print("Commands:")
    print("  python scripts/train_volleyball.py \\")
    print("    --data coco128.yaml \\")
    print("    --epochs 5 \\")
    print("    --batch 4 \\")
    print("    --model-size s")
    print()
    
    print_section("After Training")
    
    print("Once training completes, test your model:\n")
    print("  python scripts/inference.py \\")
    print("    --model runs/detect/volleyball/weights/best.pt \\")
    print("    --video test_video.mp4 \\")
    print("    --track \\")
    print("    --classify-teams\n")
    
    print_section("Important Notes")
    
    print("💡 The system works WITHOUT training (zero-training mode)")
    print("   Fine-tuning improves accuracy by ~10-15%\n")
    print("🎯 Recommended for beginners: Option 1 (Roboflow)")
    print("🎯 Recommended for custom data: Option 2 (Annotation)")
    print("🎯 Recommended for testing: Option 3 (Quick test)\n")
    
    print_section("Quick Links")
    
    print("📖 Full training guide: TRAINING.md")
    print("📝 Annotation guide: ANNOTATION_GUIDE.md")
    print("🚀 Quick start: QUICKSTART.md")
    print("📚 Full documentation: README.md\n")
    
    print("=" * 60)
    print("  Ready to start training!")
    print("=" * 60)


if __name__ == "__main__":
    demo_workflow()
