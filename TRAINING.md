# 🏐 Volleyball Training - Quick Start

Complete guide to train your own volleyball detection model.

---

## 🎯 Three Training Options

### **Option 1: Download Roboflow Dataset** ⭐ Recommended

Fastest way to get quality training data.

```bash
# Activate environment
.\venv\Scripts\activate

# Download dataset (need free API key from https://app.roboflow.com/settings/api)
cd datasets\roboflow
python download_datasets.py --api-key YOUR_API_KEY

# Train (1-2 hours on GPU)
cd ..\..
python scripts\train_volleyball.py
```

**Time:** ~15 min setup + 1-2 hours training

---

### **Option 2: Annotate Your Own Videos**

Custom training data from your volleyball footage.

```bash
# Step 1: Extract frames from video
python scripts\extract_frames.py --video your_video.mp4 --output datasets\volleyball\images\train

# Step 2: Annotate frames
#   → Use CVAT: https://app.cvat.ai
#   → Or LabelImg: pip install labelImg
#   → See ANNOTATION_GUIDE.md for details

# Step 3: Export labels to datasets\volleyball\labels\train\

# Step 4: Split train/val (80/20)
#   → Move 20% of images+labels to val/ folders

# Step 5: Train
python scripts\train_volleyball.py
```

**Time:** 3-5 hours annotation + 1-2 hours training

---

### **Option 3: Quick Test Training**

Test the pipeline without real data.

```bash
# Use small COCO subset for testing
python scripts\train_volleyball.py --data coco128.yaml --epochs 5 --batch 4 --model-size s
```

**Time:** ~15 minutes

---

## 📊 Expected Results

After training, you should see:

| Metric | Target Value |
|--------|--------------|
| mAP@0.5 (player) | > 0.85 |
| mAP@0.5 (ball) | > 0.70 |
| Precision | > 0.80 |
| Recall | > 0.75 |

---

## 🚀 Training Commands

### Basic Training (Recommended)
```bash
python scripts\train_volleyball.py
```

### Fast Training (Testing)
```bash
python scripts\train_volleyball.py --model-size s --epochs 10 --batch 4
```

### High Accuracy (Slow)
```bash
python scripts\train_volleyball.py --model-size l --epochs 50 --batch 4
```

### Resume Training
```bash
python scripts\train_volleyball.py --resume
```

---

## 📁 What You Need

**Minimum dataset structure:**

```
datasets/volleyball/
├── images/
│   ├── train/       # 300-500 images
│   └── val/         # 75-125 images
├── labels/
│   ├── train/       # YOLO format .txt files
│   └── val/
└── data.yaml        # Already created ✅
```

---

## ⚙️ Training Settings

The `train_volleyball.py` script uses **volleyball-optimized settings**:

- **Model:** YOLOv8m (best speed/accuracy balance)
- **Image size:** 960px (broadcast resolution)
- **Epochs:** 30 (with early stopping)
- **Augmentation:**
  - ✅ Color jitter (lighting variations)
  - ✅ Horizontal flip
  - ✅ Scale/zoom
  - ❌ No rotation (players upright)
  - ❌ No perspective (fixed camera)

---

## 💻 Hardware Requirements

| GPU | Batch Size | Training Time (30 epochs) |
|-----|------------|---------------------------|
| RTX 4090 | 16 | ~45 min |
| RTX 3080 | 8 | ~1.5 hours |
| RTX 3060 | 4 | ~2.5 hours |
| CPU | 2 | ~8+ hours ⚠️ |

**Out of memory?**
```bash
python scripts\train_volleyball.py --batch 4 --model-size s
```

---

## 📈 Monitor Training

Training generates real-time plots:

```
runs/detect/volleyball/
├── weights/
│   ├── best.pt      # Best weights (use this!)
│   └── last.pt      # Latest weights
├── results.csv      # Metrics per epoch
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── ...
```

**Watch for:**
- ✅ mAP increasing
- ✅ Loss decreasing
- ⚠️ Overfitting (train mAP >> val mAP)

---

## 🧪 Test Your Model

After training:

```bash
# Test on video
python scripts\inference.py `
  --model runs\detect\volleyball\weights\best.pt `
  --video test_video.mp4 `
  --track `
  --classify-teams

# Compare with COCO baseline
python scripts\inference.py `
  --model yolov8m.pt `
  --video test_video.mp4
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | `--batch 4 --model-size s` |
| **No dataset** | Download Roboflow or annotate frames |
| **Low accuracy** | More training data, longer training |
| **Slow training** | Use smaller model or reduce image size |
| **No GPU** | Add `--device cpu` (very slow) |

---

## 📝 Full Workflow Example

```powershell
# 1. Setup (if not done)
cd Volleyball
.\venv\Scripts\activate

# 2. Get training data (choose one)

# Option A: Roboflow
cd datasets\roboflow
python download_datasets.py --api-key YOUR_KEY
cd ..\..

# Option B: Custom video
python scripts\extract_frames.py --video my_game.mp4
# → Annotate frames with CVAT
# → Export to datasets/volleyball/

# 3. Train
python scripts\train_volleyball.py

# 4. Test
python scripts\inference.py `
  --model runs\detect\volleyball\weights\best.pt `
  --video test.mp4 `
  --track `
  --classify-teams

# 5. Use in production!
```

---

## 💡 Tips

1. **Start with Roboflow** - quickest path to results
2. **Use GPU** - CPU training is very slow
3. **Monitor early** - stop if not improving
4. **Test on real footage** - validate accuracy
5. **Zero-training works** - fine-tuning is optional!

---

**Ready?** Start with: `python scripts\train_volleyball.py`
