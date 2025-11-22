# Plant Segmentation - Quick Start Guide

## What's New?

Your web app now has a **Plant Segmentation** feature powered by the PlantSeg AI model. Upload multiple plant images and get instant analysis of plant organs (leaves, stems, roots).

## Quick Start (5 minutes)

### Step 1: Start the Web App
```bash
python web_app.py
```

You should see:
```
🌿 Agricultural AI Platform - Starting...
📊 Access the web app at: http://localhost:5000
✓ PlantSeg model loaded successfully
```

### Step 2: Open in Browser
Go to: `http://localhost:5000`

### Step 3: Access Plant Segmentation
**Option A (Recommended):**
- Click on the "🔬 Plant Segmentation" tab
- Click the blue "Start Segmentation" button

**Option B (Direct URL):**
- Navigate to: `http://localhost:5000/plant-segmentation`

### Step 4: Upload Images
1. **Drag & Drop**: Drag plant images onto the upload area
2. **Or Click "Choose Images"** button to select files
3. Select up to 10 images at once

**Supported formats:** JPEG, PNG, BMP, TIFF
**Max file size:** 16MB total per upload

### Step 5: Analyze
Click the **"Start Segmentation"** button

Watch the spinner - typically takes 1-10 seconds depending on:
- Number of images
- If using GPU (faster) or CPU (slower)

### Step 6: View Results
For each image you'll see:

📸 **Original Image**
- Overlaid with colored segmentation mask

📊 **Statistics**
- Confidence score (how confident the model is)
- Number of regions detected

📈 **Organ Breakdown**
Shows percentage breakdown:
- **Background** (black): Pixels outside plant
- **Plant/Leaves** (green): Leaf tissue
- **Stem** (brown): Plant stem
- **Roots** (red): Root system

## Understanding Results

### Confidence Score
- **90-100%**: Very high confidence (trust these results)
- **80-90%**: High confidence
- **70-80%**: Moderate confidence
- **<70%**: Low confidence (may need manual review)

### Class Breakdown Example
```
Organ Breakdown:
Background: 45%      ← Empty space around plant
Plant/Leaves: 36%    ← Leaf area
Stem: 15%            ← Plant stem
Roots: 4%            ← Root system
```

## Tips for Best Results

✅ **Do:**
- Use clear, well-lit photos
- Include entire plant in frame
- Use simple, contrasting backgrounds
- Take photos from directly above
- Use consistent lighting

❌ **Avoid:**
- Blurry images
- Extreme angles
- Shadows across the plant
- Very dark or overexposed photos
- Mixed crops (multiple plants together)

## Common Issues & Solutions

### Issue: "Model not available" error
**Solution:**
1. Check server logs for error messages
2. Ensure PlantSeg folder exists and contains model configs
3. Restart the Flask app

### Issue: Processing is slow
**Solution:**
1. If using CPU, consider using GPU (if available)
2. Process fewer images at once (≤5 instead of 10)
3. Reduce image size before uploading (crop to plant)

### Issue: Inaccurate segmentation
**Possible causes:**
- Poor image quality or lighting
- Plant partially cut off in frame
- Unusual plant variety not seen in training data
- Model limitations (it's trained on specific plant types)

**Solutions:**
- Try with clearer images
- Include full plant in frame
- Use well-lit, natural lighting
- For custom plants, consider fine-tuning the model

## API Usage (Developers)

### Endpoint: POST `/api/segment-plants`

**Request:**
```bash
curl -X POST http://localhost:5000/api/segment-plants \
  -F "images=@plant1.jpg" \
  -F "images=@plant2.jpg"
```

**Response:**
```json
{
  "total_images": 2,
  "processed": 2,
  "results": [
    {
      "filename": "plant1.jpg",
      "status": "success",
      "visualization": "data:image/png;base64,...",
      "segmentation_stats": {
        "0": {"count": 15000, "percentage": 45.5},
        "1": {"count": 12000, "percentage": 36.4},
        "2": {"count": 5000, "percentage": 15.2},
        "3": {"count": 1200, "percentage": 2.9}
      },
      "class_labels": {
        "0": "Background",
        "1": "Plant/Leaves",
        "2": "Stem",
        "3": "Roots"
      },
      "confidence": 0.95
    }
  ]
}
```

### Check Model Status: GET `/api/segment-status`

```bash
curl http://localhost:5000/api/segment-status
```

Returns model capabilities and current status.

## Key Features

| Feature | Details |
|---------|---------|
| **Batch Processing** | Upload 1-10 images at once |
| **AI Model** | DeepLabV3 + ResNet101 (128×128 input) |
| **Classes** | 4 (Background, Leaves, Stem, Roots) |
| **Speed** | ~0.5-1s per image (with GPU) |
| **GPU Support** | Auto-detects CUDA/GPU availability |
| **Visualization** | Color-coded overlay of segmentation |
| **Export** | Results as JSON from API |

## What the Model Does

The PlantSeg model is a **semantic segmentation** AI that:

1. **Takes** a plant image as input
2. **Analyzes** each pixel to determine which plant organ it belongs to
3. **Outputs** a segmentation mask showing organ boundaries
4. **Returns** statistics about each organ's size/coverage

Think of it as "coloring in" different parts of the plant with different colors to show where each organ is.

## File Structure

```
Plant Segmentation System:
├── web_app.py              ← Flask routes
├── plantseg_inference.py   ← Model interface (NEW)
├── templates/
│   └── plant_segmentation.html  ← Web interface (NEW)
└── PlantSeg/               ← AI model & configs
```

## Python API (for developers)

```python
from plantseg_inference import get_inferencer
import numpy as np

# Get the model
model = get_inferencer()

# Load an image
image = model.load_image_from_file('path/to/plant.jpg')

# Segment it
result = model.segment_image(image)

# Get results
mask = result['segmentation_mask']
confidence = result['confidence_map']

# Visualize
vis = model.visualize_segmentation(image, mask)
```

## Next Steps

1. **Test** with sample images from `data/plantsegv3/images/`
2. **Experiment** with different plant types
3. **Save** results (screenshot from browser or use API)
4. **Integrate** results into your farm management system

## Support

For issues:
1. Check server console for error messages
2. Review `PLANTSEG_INTEGRATION_GUIDE.md` for detailed docs
3. Test with sample images first
4. Check browser console (F12) for frontend errors

---

**Happy segmenting! 🌿**
