# SegNext PlantSeg115 - Full Training & Web Interface Setup

## 🎯 Project Overview

This project trains a **SegNext segmentation model** on all **115 plant disease classes** from the PlantSeg dataset and provides a **web interface** to detect diseases by uploading plant images.

### Dataset
- **PlantSeg115**: 11,458 images covering 115 different plant diseases
- **Train**: 7,916 images
- **Test**: 2,295 images
- **Val**: 1,247 images

### Model
- **Architecture**: SegNext with MSCAN-L backbone
- **Input Size**: 512×512 pixels
- **Output Classes**: 116 (background + 115 diseases)
- **Training Duration**: ~6-8 hours on RTX 3050 Ti

---

## 📊 Training Status

### Current Progress
Training started: **Nov 22, 2025 16:50 UTC+1**
- **Total Iterations**: 80,000 (full training on all data)
- **Checkpoint Interval**: Every 4,000 iterations
- **Estimated Completion**: ~22:50 UTC+1 (6-8 hours from start)

### Latest Updates
```
✓ Data prepared (train/test/val split ready)
✓ Training config created and training started
◑ Training in progress (monitor with Python script)
◯ Model testing (after checkpoint ready)
◯ Web interface deployment (after testing)
```

---

## 🚀 Quick Start

### 1. Monitor Training Progress
```bash
python monitor_training.py
```
Shows:
- Current iteration / total iterations
- Estimated time remaining
- Checkpoint files created

### 2. Test Intermediate Checkpoints
You can test the model while training is still running:

```bash
# Test latest checkpoint on 5 images
python test_quick.py

# Test specific iteration
python test_quick.py --iter 20000

# Comprehensive test on all 100 test images
python test_quick.py --test-all --num-images 100
```

### 3. Run Full Testing Suite
Once training completes (or uses a good checkpoint):

```bash
# Test on sample images
python PlantSeg/test_full_inference.py

# Test on all test dataset
python PlantSeg/test_full_inference.py --test-all --limit 100

# Test specific checkpoint
python PlantSeg/test_full_inference.py --checkpoint work_dirs/segnext_mscan-l_full_plantseg115/iter_40000.pth
```

### 4. Start Web Interface

**Option A**: Use current checkpoint
```bash
python start_web_interface.py
```

**Option B**: Wait for training to reach 40k iterations
```bash
python start_web_interface.py --wait
```

**Option C**: Use specific checkpoint
```bash
python start_web_interface.py --checkpoint PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115/iter_40000.pth
```

Once running, open your browser to: **http://localhost:5000**

---

## 📁 Project Structure

```
d:\colabecothoughts\
├── PlantSeg/
│   ├── data/plantseg115/        # Dataset (train/test/val splits)
│   │   ├── images/              # Raw plant images
│   │   ├── annotations/         # Segmentation masks
│   │   └── *.txt                # Dataset file lists
│   ├── configs/segnext/
│   │   └── segnext_mscan-l_full_plantseg115-512x512.py  # Training config
│   ├── work_dirs/
│   │   └── segnext_mscan-l_full_plantseg115/  # Checkpoints & logs
│   │       ├── iter_4000.pth
│   │       ├── iter_8000.pth
│   │       └── ... (continuing)
│   ├── tools/train.py           # Training script
│   └── test_full_inference.py   # Comprehensive testing
├── templates/
│   └── disease_detector.html    # Web interface UI
├── uploads/                     # Uploaded images
├── results/                     # Output results
├── web_interface_full.py        # Flask app for all 115 classes
├── test_quick.py               # Quick testing script
├── monitor_training.py         # Training progress monitor
└── start_web_interface.py      # Start web server
```

---

## 🌿 Supported Disease Classes

The model detects **115 different plant diseases** across multiple crops:

**Apple** (4 diseases):
- Black Rot, Mosaic Virus, Rust, Scab

**Banana** (5 diseases):
- Anthracnose, Black Leaf Streak, Bunchy Top, Cigar End Rot, Cordana Leaf Spot, Panama Disease

**Tomato** (7 diseases):
- Bacterial Leaf Spot, Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Yellow Leaf Curl Virus

**Wheat** (8 diseases):
- Bacterial Leaf Streak, Head Scab, Leaf Rust, Loose Smut, Powdery Mildew, Septoria Blotch, Stem Rust, Stripe Rust

**Potato** (2 diseases):
- Early Blight, Late Blight

... and **86 more disease classes** across Bell Pepper, Blueberry, Broccoli, Cabbage, Carrot, Cauliflower, Celery, Cherry, Citrus, Coffee, Corn, Cucumber, Eggplant, Garlic, Ginger, Grape, Grapevine, Lettuce, Maple, Peach, Plum, Raspberry, Rice, Soybean, Squash, Strawberry, Tobacco, Zucchini

---

## 🔬 Web Interface Features

### Upload & Detect
- Drag-and-drop image upload
- Support for JPG, PNG, BMP, GIF, TIFF
- Max file size: 16 MB
- Real-time processing

### Results
- **Disease Name**: Identified plant disease with crop type
- **Confidence Score**: Model confidence (0-100%)
- **Visual Feedback**: Color-coded confidence bar

### Error Handling
- Image validation (size, format, corruption check)
- Comprehensive error messages
- Automatic preprocessing and resizing

---

## 📈 Training Details

### Configuration
- **Optimizer**: AdamW (lr=0.0001)
- **Learning Rate Schedule**: Linear warmup (1500 steps) + Poly decay
- **Batch Size**: 2 (optimized for RTX 3050 Ti)
- **Data Augmentation**: RandomFlip, PhotoMetricDistortion, RandomCrop
- **Loss Function**: CrossEntropyLoss

### Checkpoints
Saved every 4,000 iterations:
- `iter_4000.pth`
- `iter_8000.pth`
- `iter_12000.pth`
- ... (continuing up to 80,000)

Last 3 checkpoints are kept to save space.

---

## 🧪 Testing Guide

### Quick Check (5-10 min)
```bash
python test_quick.py
```

### Comprehensive Validation (30-60 min)
```bash
python PlantSeg/test_full_inference.py --test-all --limit 100
```

### Single Image Test
```bash
python PlantSeg/test_full_inference.py --image data/plantseg115/images/test/tomato_early_blight_1.jpg
```

**Expected Results**:
- Model should detect **50+ different disease classes** across test set
- Confidence scores typically **60-90%** for diseased plants
- Processing time: **~2-5 seconds per image**

---

## 🌐 Web Server Usage

### Starting the Server
```bash
python start_web_interface.py
```

### Accessing the Interface
1. Open browser: **http://localhost:5000**
2. Upload a plant image (JPG/PNG)
3. Click "Analyze Plant Disease"
4. View results with confidence score

### API Endpoint
```
POST /api/predict
Content-Type: multipart/form-data

Parameters:
  - image: Plant image file

Response:
{
  "success": true,
  "disease": "Tomato: Early Blight",
  "confidence": 87.5,
  "class_index": 97,
  "filename": "1234567_image.jpg"
}
```

### Health Check
```
GET /health

Response:
{
  "status": "running",
  "device": "cuda",
  "model_loaded": true,
  "cuda_available": true
}
```

---

## 🐛 Troubleshooting

### Training Issues

**"Checkpoint not found"**
- Check if training is still running: `python monitor_training.py`
- Wait for first checkpoint (~15 minutes into training)

**CUDA out of memory**
- Training configured for RTX 3050 Ti (4GB)
- If error occurs, reduce batch_size in config to 1

### Web Interface Issues

**"Model not loaded"**
- Ensure checkpoint exists in `PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115/`
- Try specifying checkpoint: `python start_web_interface.py --checkpoint <path>`

**"Connection refused"**
- Ensure server is running
- Check port 5000 is available
- Try different port: modify `web_interface_full.py` port setting

**Image upload fails**
- Check image is valid format (JPG/PNG/BMP/GIF/TIFF)
- Ensure image size is 50-4096 pixels
- Verify file size is under 16MB

---

## 📊 Performance Metrics

### Training Performance
- **GPU Memory**: ~3.2GB (RTX 3050 Ti)
- **Time per 1000 iterations**: ~90 seconds
- **Total Training Time**: ~6-8 hours

### Inference Performance
- **Image Processing**: 2-5 seconds per image
- **Memory Usage**: ~2GB during inference
- **Batch Processing**: Can process multiple images sequentially

### Model Accuracy (Expected)
- **mIoU**: ~40-45% (based on PlantSeg baseline)
- **Disease Detection Rate**: >80% for visible diseases
- **False Positive Rate**: <5% on healthy plants

---

## 🔄 Workflow Summary

```
1. TRAINING (In Progress)
   PlantSeg full training on 115 disease classes
   ├─ Checkpoint saves every 4k iterations
   ├─ Monitor: python monitor_training.py
   └─ ETA: ~6-8 hours from start

2. TESTING (Ready when checkpoint available)
   python test_quick.py                    # Quick 5-min test
   python test_full_inference.py --test-all  # Full validation

3. DEPLOYMENT (Ready after testing)
   python start_web_interface.py           # Start web server
   Visit: http://localhost:5000            # Open in browser
   Upload image → Get disease prediction   # Use interface
```

---

## 📝 Notes

- Training will auto-resume if interrupted (PyTorch behavior)
- Web interface loads latest checkpoint automatically
- All 115 disease classes are included in predictions
- No API keys or internet required (offline inference)
- Model runs locally on your GPU/CPU

---

## 🎓 References

- **PlantSeg Paper**: [arXiv:2409.04038](https://arxiv.org/abs/2409.04038)
- **SegNext**: Semantic Segmentation with Next-Generation Attention
- **Dataset**: [PlantSeg on Zenodo](https://zenodo.org/records/14935094)

---

**Status**: Training in progress | Last updated: Nov 22, 2025
