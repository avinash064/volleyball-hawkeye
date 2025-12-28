# Model Weights Setup

## Download RT-DETR Weights

The trained RT-DETR model for volleyball detection is available on Google Drive:

**Download Link**: [RT-DETR Volleyball Weights](https://drive.google.com/file/d/1uASPCHAk6kDuVV8eZnTzEUx-zccD4-ik/view?usp=sharing)

### File Details
- **File Name**: `best(2).pt`
- **Model**: RT-DETR (Real-Time Detection Transformer)
- **Classes**: player, referee, volleyball
- **Framework**: Ultralytics

### Installation Steps

1. **Download the weights file** from the Google Drive link above

2. **Place the file** in one of these locations:
   ```
   C:\Users\xghostrider\Downloads\best(2).pt
   ```
   OR in the project root:
   ```
   Volleyball/weights/best(2).pt
   ```

3. **Update the path** in the Python scripts:

   For `hawkeye_complete.py`:
   ```python
   WEIGHTS = r"path/to/best(2).pt"
   ```

   For `batch_process.py`:
   ```python
   WEIGHTS_PATH = r"path/to/best(2).pt"
   ```

### Verify Installation

Run this command to verify the weights are accessible:

```python
from ultralytics import RTDETR
model = RTDETR("path/to/best(2).pt")
print("✓ Weights loaded successfully")
```

### Alternative: Direct Download via gdown

You can also download using `gdown`:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1uASPCHAk6kDuVV8eZnTzEUx-zccD4-ik
```

### Training Your Own Weights

If you want to train your own RT-DETR model:

1. Prepare volleyball dataset (YOLO format)
2. Use Ultralytics training:
   ```python
   from ultralytics import RTDETR
   
   model = RTDETR('rtdetr-l.pt')
   model.train(data='volleyball.yaml', epochs=100)
   ```

3. Export trained weights:
   ```python
   model.export(format='pt')
   ```

### Troubleshooting

**Issue**: `FileNotFoundError: best(2).pt`
- Verify the file path
- Check file permissions
- Ensure the file is not corrupted

**Issue**: `Model loading error`
- Update ultralytics: `pip install --upgrade ultralytics`
- Verify PyTorch compatibility
- Check CUDA availability

### Model Performance

The provided weights achieve:
- **Player Detection**: High accuracy on volleyball broadcasts
- **Ball Detection**: Optimized for fast-moving volleyball
- **Referee Detection**: Distinguishes from players
- **Inference Speed**: ~5-8 FPS on 1080p video (GPU)

---

For questions about the model weights, open a GitHub issue.
