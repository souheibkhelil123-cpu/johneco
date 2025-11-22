# Plant Disease Detector - Web Interface

A beautiful web application that uses AI to detect plant diseases from uploaded images. Upload a plant image and get instant disease identification powered by your trained segmentation model.

## Features

✨ **User-Friendly Interface**
- Drag-and-drop image upload
- Real-time image preview
- Beautiful gradient design with smooth animations

🤖 **AI-Powered Detection**
- 114 different plant disease classes
- Deep learning segmentation model (MSCAN backbone)
- Confidence score for each prediction

📱 **Responsive Design**
- Works on desktop, tablet, and mobile
- Touch-friendly upload area
- Mobile-optimized UI

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r web_requirements.txt
```

### 2. Verify Your Model

Make sure your trained model is in the correct location:
```
PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth
```

### 3. Run the Web Server

```bash
python web_interface.py
```

You should see output like:
```
============================================================
PLANT DISEASE DETECTOR - WEB SERVER
============================================================
Device: cuda
✓ Model loaded successfully!

🚀 Starting Flask server...
Visit http://localhost:5000 in your browser
============================================================
```

### 4. Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

## How to Use

1. **Upload Image**
   - Click the upload area or drag-and-drop an image
   - Supported formats: JPG, PNG, BMP (Max 16MB)

2. **Preview**
   - Your image appears in the preview
   - Adjust if needed, or upload a different image

3. **Analyze**
   - Click "🔍 Analyze Plant Disease" button
   - Wait for the AI model to process (a few seconds)

4. **View Result**
   - Disease name displayed prominently
   - Confidence percentage with visual bar
   - Click "🔄 Analyze Another Image" to try again

## Supported Diseases (114 Classes)

The model can detect diseases across:
- **Apple** (4 diseases): Black Rot, Mosaic Virus, Rust, Scab
- **Banana** (6 diseases): Anthracnose, Black Leaf Streak, Bunchy Top, Cigar End Rot, Cordana Leaf Spot, Panama Disease
- **Basil, Bean, Bell Pepper, Blueberry, Broccoli, Cabbage, Carrot, Cauliflower, Celery, Cherry, Citrus, Coffee, Corn, Cucumber, Eggplant, Garlic, Ginger, Grape, Grapevine, Lettuce, Maple, Peach, Plum, Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tobacco, Tomato, Wheat, Zucchini**

## API Endpoints

### `GET /`
Serves the HTML interface

### `POST /api/predict`
Send an image for disease prediction

**Request:**
```
POST /api/predict
Content-Type: multipart/form-data

image: <image_file>
```

**Response:**
```json
{
    "success": true,
    "disease": "Tomato: Tomato Early Blight",
    "confidence": 87.34,
    "class_index": 97,
    "filename": "image.jpg"
}
```

### `GET /health`
Check server status

**Response:**
```json
{
    "status": "running",
    "device": "cuda",
    "model_loaded": true
}
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the Flask app:
```python
app.run(debug=False, host='0.0.0.0', port=5001)  # Change to port 5001
```

### Model Not Found
Make sure the model checkpoint exists:
```bash
# Check if file exists
ls PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth
```

If missing, train the model first:
```bash
cd PlantSeg
bash run.sh
```

### CUDA/GPU Issues
To use CPU instead of GPU:
1. Edit `web_interface.py`
2. Change: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
3. Or force CPU: `device = 'cpu'`

### Slow Predictions
- First prediction is slower (model loading)
- Subsequent predictions are faster
- Consider GPU acceleration if available

## Performance

- **Model Size**: ~100MB
- **Inference Time**: 1-3 seconds per image (GPU)
- **Memory Usage**: ~4GB GPU VRAM required
- **Max Image Size**: 16MB

## Browser Compatibility

✅ Chrome/Edge (v90+)
✅ Firefox (v88+)
✅ Safari (v14+)
✅ Mobile browsers

## Customization

### Change Port
Edit `web_interface.py`:
```python
app.run(debug=False, host='0.0.0.0', port=8000)  # Use port 8000
```

### Allow External Access
Edit `web_interface.py`:
```python
app.run(debug=False, host='0.0.0.0', port=5000)  # Already allows all IPs
```

### Adjust Upload Size Limit
Edit `web_interface.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

## Production Deployment

For production deployment (e.g., with Gunicorn):

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 web_interface:app
```

**Note:** Single worker (`-w 1`) recommended for GPU usage

## Development

To run with debug mode:

```python
# In web_interface.py, change:
app.run(debug=True, host='0.0.0.0', port=5000)
```

Then run:
```bash
python web_interface.py
```

Changes to HTML/CSS will auto-reload. Python changes require manual restart.

## License

This project uses the PlantSeg model and MMSegmentation framework.

## Support

For issues or questions:
1. Check server logs in terminal
2. Open browser console (F12) for client errors
3. Verify model file exists
4. Check system GPU memory availability
