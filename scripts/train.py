"""
Training Script for Volleyball Detection Model

Fine-tune YOLOv8 on volleyball-specific datasets.
Can be skipped if using zero-training fallback with COCO weights.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train volleyball detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/volleyball/data.yaml",
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Pretrained model to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size"
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
        help="Device to train on (e.g., '0' for GPU 0, 'cpu')"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="volleyball",
        help="Experiment name"
    )
    return parser.parse_args()


def train(args):
    """Run training"""
    
    print("=" * 60)
    print("Volleyball Detection Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load pretrained model
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        
        # Optimizer
        optimizer="AdamW",
        lr0=0.0008,
        warmup_epochs=3,
        cos_lr=True,
        
        # Early stopping
        patience=10,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=3,
        translate=0.1,
        scale=0.5,
        mosaic=0.5,
        mixup=0.1,
        
        # Logging
        verbose=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best weights saved to: runs/detect/{args.name}/weights/best.pt")
    print(f"Last weights saved to: runs/detect/{args.name}/weights/last.pt")
    print("\nNext steps:")
    print("  1. Review training plots and metrics")
    print("  2. Run inference: python scripts/inference.py --model runs/detect/{}/weights/best.pt".format(args.name))
    print("  3. Evaluate on validation set: python scripts/evaluate.py")


def main():
    args = parse_args()
    
    # Validate data config exists
    if not Path(args.data).exists():
        print(f"❌ Data config not found: {args.data}")
        print("\nPlease ensure:")
        print("  1. Dataset is downloaded and organized")
        print("  2. data.yaml is properly configured")
        print("  3. Image and label paths are correct")
        return
    
    train(args)


if __name__ == "__main__":
    main()
