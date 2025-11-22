# Project Structure - Plant Disease Detector

```
colabecothoughts/
│
├── 🌐 WEB INTERFACE (NEW)
│   ├── web_interface.py              # Flask backend server
│   ├── templates/
│   │   └── disease_detector.html     # Beautiful web UI
│   ├── uploads/                      # Uploaded images (auto-created)
│   ├── web_requirements.txt          # Dependencies for web server
│   ├── WEB_INTERFACE_README.md       # Full documentation
│   └── QUICK_START.md                # Quick setup guide
│
├── 🤖 AI MODEL (TRAINED)
│   ├── PlantSeg/
│   │   ├── work_dirs/
│   │   │   └── segnext_mscan-l_test/
│   │   │       ├── iter_1000.pth     # ✅ Trained checkpoint (1000 iterations)
│   │   │       └── latest.pth        # Symlink to latest
│   │   ├── configs/
│   │   │   └── segnext/
│   │   │       └── segnext_simple_256.py  # Config (256x256, fixed crop)
│   │   ├── data/
│   │   │   └── plantseg115/
│   │   │       ├── Metadatav2.csv   # Class definitions
│   │   │       ├── images/
│   │   │       ├── annotations/
│   │   │       └── annotation_*.json
│   │   ├── tools/
│   │   │   └── train.py             # Training script
│   │   ├── predict_disease.py       # Standalone prediction
│   │   ├── test_inference.py        # Inference test script
│   │   └── run.sh                   # Training launcher
│   │
│   ├── 📊 RESULTS & OUTPUTS
│   ├── results/
│   │   ├── predicted_disease.jpg    # Output with label
│   │   ├── segmentation_map.png     # Color segmentation
│   │   └── original_input.jpg       # Input image
│   │
│   └── 📚 DEPENDENCIES
│       ├── mmsegmentation/          # Framework
│       ├── requirements.txt
│       └── PlantSeg/requirements.txt
│
├── 🧪 TESTING & UTILITIES
│   ├── TEST_CROP_RECOMMENDATION.py  # Crop analysis
│   ├── plantseg_inference.py        # Inference utilities
│   ├── test_model.py                # Model validation
│   ├── testimage.jpg                # ✅ Sample test image
│   └── __pycache__/                 # Compiled Python cache
│
├── 📖 DOCUMENTATION
│   ├── README.md                     # Main documentation
│   ├── ARCHITECTURE.md               # System design
│   ├── QUICK_START.md               # 🆕 Quick start guide
│   ├── WEB_INTERFACE_README.md      # 🆕 Web interface docs
│   └── requirements.txt
│
└── 🎯 CONFIG FILES
    ├── pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py
    ├── pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth
    └── web_app.py                   # Original web app (legacy)
```

## 🚀 Quick Navigation

### To Use the Web Interface
```bash
cd d:/colabecothoughts
python web_interface.py
# Then open: http://localhost:5000
```

### To Train the Model
```bash
cd d:/colabecothoughts/PlantSeg
bash run.sh
```

### To Run Standalone Inference
```bash
cd d:/colabecothoughts/PlantSeg
python predict_disease.py
```

## 📊 Model Specifications

| Aspect | Details |
|--------|---------|
| **Framework** | PyTorch + MMSegmentation |
| **Architecture** | MSCAN-L (Multi-Scale Convolutional Attention Network) |
| **Decoder** | LightHamHead |
| **Input Size** | 256×256 pixels (fixed) |
| **Output Classes** | 114 (plant diseases) |
| **Dataset** | PlantSeg115 (116 classes total) |
| **Training Iterations** | 1000 (no validation, avoids OOM) |
| **Optimizer** | AdamW (lr=0.0001) |
| **Batch Size** | 1 (memory-constrained) |
| **GPU Memory** | ~1.2 GB average usage |

## 🎯 Web Interface Features

| Feature | Implementation |
|---------|-----------------|
| **Upload** | Drag-and-drop or click |
| **Preview** | Real-time image preview |
| **Processing** | GPU-accelerated inference |
| **Display** | Disease name + confidence % |
| **Response Time** | 2-3 seconds per image |
| **Responsive** | Mobile/tablet/desktop compatible |

## 📈 Disease Classes (114 Total)

Organized by plant type:
- 🍎 **Apple**: Black Rot, Mosaic Virus, Rust, Scab
- 🍌 **Banana**: Anthracnose, Black Leaf Streak, Bunchy Top, Cigar End Rot, Cordana Leaf Spot, Panama Disease
- 🍅 **Tomato**: Bacterial Leaf Spot, Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Yellow Leaf Curl Virus
- 🌾 **Wheat**: Bacterial Leaf Streak, Head Scab, Leaf Rust, Loose Smut, Powdery Mildew, Septoria Blotch, Stem Rust, Stripe Rust
- And 27 more plant types (Corn, Potato, Pepper, Cucumber, Bean, Carrot, etc.)

## 🔧 Configuration Files

### Training Config
`PlantSeg/configs/segnext/segnext_simple_256.py`
- No base inheritance (avoid conflicts)
- Fixed 256×256 crops
- Batch size = 1
- No validation (prevents OOM at iter 2024)
- Simple deterministic pipeline

### Web Server Config
Directly in `web_interface.py`:
- Port: 5000
- Device: CUDA (GPU) or CPU fallback
- Max upload: 16MB
- Model path: `PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth`

## 🛠️ Troubleshooting Guides

See: **WEB_INTERFACE_README.md** → Troubleshooting section

Common issues:
- Port already in use → Change port in code
- Model not found → Train first with `bash run.sh`
- CUDA errors → Use CPU mode
- Slow predictions → GPU provides speedup

## 📦 Dependencies Installed

```bash
# ML/AI Stack
torch==2.1.0+cu121
torchvision==0.16.0+cu121
mmengine==0.10.7
opencv-python==4.8.0

# Web Server
Flask==2.3.3
Werkzeug==2.3.7

# Utilities
numpy, Pillow, scipy, scikit-learn
```

## 🎯 Next Steps

1. ✅ **Web Interface Running** → Access http://localhost:5000
2. 🧪 **Test with Images** → Upload plant images to test
3. 📊 **Monitor Results** → Check accuracy and confidence
4. 🚀 **Deploy** (Optional) → Use Gunicorn for production
5. 📈 **Improve** (Optional) → Retrain with more data

## 📞 Support Resources

- **Quick Start**: `QUICK_START.md`
- **Full Docs**: `WEB_INTERFACE_README.md`
- **Architecture**: `PlantSeg/ARCHITECTURE.md`
- **Training**: `PlantSeg/QUICK_START.md`

---

**Status**: ✅ Ready for use!
**Last Updated**: November 22, 2025
**Model Checkpoint**: 1000 iterations complete
**Server**: Running on http://localhost:5000
