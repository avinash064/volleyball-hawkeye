# Volleyball Detection & Tracking - Quick Start

## 🚀 Zero-Training Setup (Fastest Path)

This system works **out of the box** with COCO-pretrained weights. No training required!

### 1. Install Dependencies

```bash
cd Volleyball
pip install -r requirements.txt
```

### 2. Run Inference

```bash
# Basic detection + tracking
python scripts/inference.py \
  --video path/to/volleyball.mp4 \
  --output outputs/results \
  --model yolov8m.pt \
  --track \
  --classify-teams
```

**That's it!** The system will:
- Auto-download YOLOv8m weights on first run
- Detect players and ball using COCO classes
- Track with ByteTrack
- Classify teams by jersey color

---

## 📊 Optional: Fine-Tuning

Only do this if you want higher accuracy and have time to annotate/download data.

### 1. Get Roboflow API Key
https://app.roboflow.com/settings/api

### 2. Download Datasets

```bash
cd datasets/roboflow
# Edit download_datasets.py and add your API key
python download_datasets.py
```

### 3. Train

```bash
cd ../..
python scripts/train.py --data datasets/volleyball/data.yaml --epochs 30
```

### 4. Inference with Fine-Tuned Model

```bash
python scripts/inference.py \
  --video path/to/volleyball.mp4 \
  --model runs/detect/volleyball/weights/best.pt \
  --track \
  --classify-teams
```

---

## 📁 Project Structure

```
Volleyball/
├── src/                  # Core modules
│   ├── detection/        # YOLO detector
│   ├── tracking/         # ByteTrack + team classifier
│   ├── projection/       # Court homography
│   └── visualization/    # Rendering utilities
├── scripts/              # Training & inference
├── datasets/             # Training data (optional)
├── models/               # YOLO weights
└── outputs/              # Results
```

---

## ❓ Troubleshooting

**CUDA out of memory?**
- Reduce batch size: `--batch 4`
- Use smaller model: `yolov8s.pt`

**Ball not detected?**
- Lower confidence: `--conf-ball 0.10`
- Check camera angle (works best with side view)

**Team classification wrong?**
- Increase fitting frames by viewing more of the video
- Check jersey contrast (similar colors cause issues)

---

**Need help?** Check the full [README.md](README.md) for detailed documentation.
