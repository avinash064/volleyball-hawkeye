"""
Quick Training Script for Volleyball Detection

Simplified training interface with presets for volleyball.
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not installed. Please install requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


def train_volleyball_model(
    data_yaml: str = "datasets/volleyball/data.yaml",
    model_size: str = "m",
    epochs: int = 30,
    batch_size: int = 8,
    image_size: int = 960,
    device: str = "0",
    name: str = "volleyball",
    resume: bool = False,
    pretrained: str = None
):
    """
    Train volleyball detection model with optimized settings.
    
    Args:
        data_yaml: Path to dataset configuration
        model_size: Model size (n, s, m, l, x) - 'm' recommended
        epochs: Number of training epochs
        batch_size: Batch size (reduce if CUDA out of memory)
        image_size: Input image size (960 for volleyball broadcast)
        device: GPU device ('0', '1', or 'cpu')
        name: Experiment name
        resume: Resume from last checkpoint
        pretrained: Path to pretrained weights (optional)
    """
    
    print("=" * 60)
    print("🏐 Volleyball Detection Training")
    print("=" * 60)
    
    # Validate data config
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        print("\nOptions:")
        print("  1. Download dataset: python datasets/roboflow/download_datasets.py")
        print("  2. Annotate custom data: see ANNOTATION_GUIDE.md")
        print("  3. Extract frames: python scripts/extract_frames.py --video your_video.mp4")
        return
    
    # Select model
    if pretrained:
        model_path = pretrained
        print(f"📦 Loading pretrained model: {pretrained}")
    else:
        model_path = f"yolov8{model_size}.pt"
        print(f"📦 Loading YOLOv8{model_size} model")
    
    print(f"📊 Dataset: {data_yaml}")
    print(f"⚙️ Settings:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch: {batch_size}")
    print(f"   - Image size: {image_size}")
    print(f"   - Device: {device}")
    print(f"   - Experiment: {name}")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Train with volleyball-optimized settings
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        name=name,
        resume=resume,
        
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.0008,            # Initial learning rate
        lrf=0.01,              # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,           # Cosine LR scheduler
        
        # Early stopping
        patience=10,           # Stop if no improvement for 10 epochs
        
        # Data augmentation (volleyball-specific)
        hsv_h=0.015,          # Hue augmentation (minimal)
        hsv_s=0.7,            # Saturation augmentation
        hsv_v=0.4,            # Value augmentation
        degrees=3,            # Only slight rotations
        translate=0.1,        # Slight translation
        scale=0.5,            # Scale augmentation for zoom
        shear=0.0,            # No shear (players upright)
        perspective=0.0,      # No perspective (broadcast camera)
        flipud=0.0,           # No vertical flip
        fliplr=0.5,           # 50% horizontal flip OK
        mosaic=0.5,           # Mosaic augmentation
        mixup=0.1,            # Mixup augmentation
        
        # Copy paste augmentation
        copy_paste=0.0,       # Disable (not suitable for sports)
        
        # Performance
        workers=8,            # Data loading workers
        
        # Logging
        verbose=True,
        plots=True,           # Generate training plots
        save=True,
        save_period=-1,       # Save only best/last
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📊 Results saved to: runs/detect/{name}/")
    print(f"🏆 Best weights: runs/detect/{name}/weights/best.pt")
    print(f"📝 Last weights: runs/detect/{name}/weights/last.pt")
    print(f"📈 Metrics: runs/detect/{name}/results.csv")
    print(f"📊 Plots: runs/detect/{name}/")
    
    print("\n🚀 Next steps:")
    print(f"  1. Review training plots in runs/detect/{name}/")
    print(f"  2. Test model:")
    print(f"     python scripts/inference.py --model runs/detect/{name}/weights/best.pt --video test.mp4")
    print(f"  3. Evaluate:")
    print(f"     python scripts/evaluate.py --model runs/detect/{name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train volleyball detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (recommended)
  python train_volleyball.py
  
  # Quick test (small model, few epochs)
  python train_volleyball.py --model-size s --epochs 10 --batch 4
  
  # High accuracy (large model, more epochs)
  python train_volleyball.py --model-size l --epochs 50
  
  # Resume training
  python train_volleyball.py --resume
  
  # Custom dataset
  python train_volleyball.py --data my_data.yaml
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/volleyball/data.yaml",
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n(ano), s(mall), m(edium), l(arge), x(large)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (reduce if GPU memory issues)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 0, 1, 2, ... or cpu"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="volleyball",
        help="Experiment name"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained weights"
    )
    
    args = parser.parse_args()
    
    train_volleyball_model(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        name=args.name,
        resume=args.resume,
        pretrained=args.pretrained
    )


if __name__ == "__main__":
    main()
