# Volleyball Dataset Annotation Guide

## 🎯 Overview

This guide helps you create high-quality training data for volleyball detection.

---

## 📋 What to Annotate

### Classes to Label

1. **player** - All volleyball players on court
   - Include both teams
   - Include players on bench if visible
   - Full body bounding box

2. **ball** - Volleyball
   - Tight bounding box around ball
   - Annotate even if blurry
   - Skip if completely occluded

3. **referee** (optional)
   - If clearly distinguishable by uniform
   - Can be classified as player and separated later

---

## 🎨 Annotation Best Practices

### Bounding Box Rules

#### ✅ DO:
- Draw **tight boxes** around objects
- Include the **full body** for players (head to feet)
- Keep **ball boxes small** (never larger than player's head)
- Annotate **all visible objects** in frame
- Use **consistent naming** (player, ball, referee)

#### ❌ DON'T:
- Skip partially visible players (annotate even if cut off)
- Include excessive background in boxes
- Annotate motion blur frames (skip them)
- Change class names (stick to: player, ball, referee)
- Annotate if object is >90% occluded

### Frame Selection

**Sample 1 frame every 10-15 frames** (~400-500 frames total)

**Include diversity:**
- Different camera angles
- Different lighting conditions
- Various player positions (serving, spiking, receiving)
- Ball in air and grounded
- Crowded and sparse scenes

**Skip frames with:**
- Extreme motion blur
- Camera transitions
- Replays (unless different angle)
- Scoreboard/audience closeups

---

## 🛠️ Annotation Tools

### Recommended: CVAT (Computer Vision Annotation Tool)

**Why CVAT:**
- Free and open source
- Web-based (no installation)
- Supports YOLO export
- Track interpolation (saves time)

**Steps:**
1. Go to https://app.cvat.ai or self-host
2. Create account (free)
3. Create new task → Upload frames
4. Add labels: player, ball, referee
5. Annotate with rectangle tool
6. Export → YOLO 1.1 format

### Alternative: LabelImg

**For offline annotation:**

```bash
pip install labelImg
labelImg datasets/volleyball/images/train datasets/volleyball/labels/train
```

- Simple desktop app
- Works offline
- Direct YOLO format output

### Alternative: Roboflow Annotate

**For team collaboration:**
- Go to https://app.roboflow.com
- Upload images
- Annotate in browser
- Auto-suggest with AI
- Export to YOLO format

---

## 📂 Expected Output Structure

After annotation, your dataset should look like:

```
datasets/volleyball/
├── images/
│   ├── train/
│   │   ├── frame_00000.jpg
│   │   ├── frame_00001.jpg
│   │   └── ...
│   └── val/
│       ├── frame_00400.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── frame_00000.txt  # YOLO format
│   │   ├── frame_00001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

### YOLO Label Format

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example (frame_00000.txt):**
```
0 0.512 0.342 0.125 0.287    # player
0 0.234 0.654 0.098 0.234    # player
1 0.456 0.123 0.023 0.034    # ball
```

**Class IDs:**
- 0 = player
- 1 = ball
- 2 = referee

**Coordinates:**
- Normalized (0.0 to 1.0)
- x_center, y_center = center of box
- width, height = box dimensions

---

## ✅ Quality Checklist

Before training, verify:

- [ ] At least 300 annotated frames
- [ ] 80/20 train/val split
- [ ] All classes represented
- [ ] Consistent class names
- [ ] YOLO format labels (.txt files)
- [ ] Labels match image names
- [ ] data.yaml is configured
- [ ] No empty label files

---

## 🚀 After Annotation

Once annotation is complete:

```bash
# Verify dataset structure
python -c "
import os
from pathlib import Path

train_imgs = len(list(Path('datasets/volleyball/images/train').glob('*.jpg')))
train_lbls = len(list(Path('datasets/volleyball/labels/train').glob('*.txt')))
val_imgs = len(list(Path('datasets/volleyball/images/val').glob('*.jpg')))
val_lbls = len(list(Path('datasets/volleyball/labels/val').glob('*.txt')))

print(f'Train: {train_imgs} images, {train_lbls} labels')
print(f'Val: {val_imgs} images, {val_lbls} labels')
print(f'Match: {train_imgs == train_lbls and val_imgs == val_lbls}')
"

# Start training
python scripts/train.py --data datasets/volleyball/data.yaml --epochs 30
```

---

## 💡 Tips for Faster Annotation

1. **Use keyboard shortcuts** (CVAT: N for next, R for rectangle)
2. **Annotate similar frames together** (same scene)
3. **Use tracking interpolation** (CVAT feature)
4. **Start with clear frames** (build confidence)
5. **Take breaks** (avoid annotation fatigue)
6. **Quality > Quantity** (300 good frames > 1000 messy frames)

---

## 📊 Expected Time Investment

| Task | Time |
|------|------|
| Frame extraction | 5-10 min |
| Setup annotation tool | 10-15 min |
| Annotate 400 frames | 2-4 hours |
| Review & fix errors | 30 min |
| Export & organize | 10 min |
| **Total** | **3-5 hours** |

**Worth it?** Fine-tuning typically improves accuracy by 10-15% over COCO baseline.

---

## ❓ FAQ

**Q: How many frames do I need?**
A: Minimum 200-300, recommended 400-500 for good results.

**Q: Should I annotate every player?**
A: Yes, annotate all visible players, even if partially cut off.

**Q: What if the ball is blurry?**
A: Annotate it anyway with your best guess for the center.

**Q: How do I handle overlapping players?**
A: Draw separate boxes for each player, overlap is OK.

**Q: Can I skip annotation and just use COCO?**
A: Yes! The system works well with COCO weights. Fine-tuning is optional.

---

**Ready to annotate?** Start with `scripts/extract_frames.py` to get your training frames!
