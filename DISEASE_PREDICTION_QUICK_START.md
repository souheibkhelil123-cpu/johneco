# Disease Prediction Feature - Quick Start

## What's New?
Your web application now includes an AI-powered plant disease prediction system! Upload a plant leaf image and get instant disease diagnosis with confidence scores and detailed analysis.

## How to Start

### Step 1: Start the Web App
```bash
cd d:\colabecothoughts
python web_app.py
```

### Step 2: Access Disease Detection
Open your browser and go to:
```
http://localhost:5000/disease-detection
```

### Step 3: Upload and Predict
1. **Drag & Drop**: Drag a plant leaf image onto the upload box
2. **Or Click**: Click the upload area to browse files
3. **Analyze**: Click the "Analyze Image" button
4. **View Results**: See disease prediction and confidence
5. **Download**: Download prediction image and segmentation map

## Features

✅ **114+ Disease Detection**
- Covers crops: Tomato, Potato, Wheat, Corn, Rice, Beans, Peppers, and more

✅ **Real-time Analysis**
- AI-powered segmentation model processes your image in seconds

✅ **Confidence Scoring**
- Know how confident the model is about its prediction

✅ **Visual Results**
- See predicted disease overlaid on your image
- Analyze segmentation map

✅ **Download Results**
- Save prediction images for records
- Export segmentation analysis

## Supported Image Formats
- PNG
- JPG / JPEG
- GIF
- BMP

## File Size Limit
Maximum 50MB per image

## API Usage

### Python
```python
import requests

with open('leaf_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict-disease',
        files={'file': f}
    )
    result = response.json()
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']}%")
```

### JavaScript
```javascript
const input = document.getElementById('fileInput');
const formData = new FormData();
formData.append('file', input.files[0]);

fetch('/api/predict-disease', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => {
    console.log('Disease:', data.disease);
    console.log('Confidence:', data.confidence + '%');
});
```

### cURL
```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:5000/api/predict-disease
```

## Example Responses

### Success
```json
{
    "success": true,
    "disease": "Tomato: Tomato Early Blight",
    "disease_class": 97,
    "confidence": 82.45,
    "predicted_image": "/results/result_123456_predicted.jpg",
    "segmentation_map": "/results/result_123456_segmentation.png"
}
```

### Error
```json
{
    "success": false,
    "error": "File type not allowed"
}
```

## Disease Categories

### Vegetables
- Tomato (7 diseases)
- Potato (2 diseases)
- Bell Pepper (4 diseases)
- Cucumber (3 diseases)
- Eggplant (3 diseases)
- And more...

### Fruits
- Apple (4 diseases)
- Banana (6 diseases)
- Grape (3 diseases)
- Peach (5 diseases)
- Strawberry (2 diseases)
- And more...

### Grains
- Wheat (8 diseases)
- Corn (4 diseases)
- Rice (2 diseases)
- Soybean (5 diseases)
- And more...

### Herbs & Others
- Basil, Bean, Blueberry, Carrot, Cauliflower, Celery, Cherry, Citrus, Coffee, Garlic, Ginger, Lettuce, Plum, Raspberry, Tobacco, Zucchini

## Integration with Crop Recommendation

The disease prediction works seamlessly with existing crop recommendation:

1. **Predict Disease** → Upload plant image
2. **Get Recommendation** → Get resistant crop varieties
3. **Analyze Conditions** → Check soil/environmental requirements
4. **Implement Treatment** → Follow disease management tips

## Tips for Best Results

1. **Clear Images**: Use well-lit, clear photos of affected leaves
2. **Close-up**: Get close enough to see disease symptoms clearly
3. **Multiple Angles**: Try different photos if initial prediction seems off
4. **Background**: Plain background works best
5. **Size**: Image doesn't need to be large (512x512 optimal)

## Browser Compatibility
- ✅ Chrome/Edge (Latest)
- ✅ Firefox (Latest)
- ✅ Safari (Latest)
- ✅ Mobile browsers

## Performance

**Typical Response Times**:
- First upload: 3-5 seconds (model loading)
- Subsequent uploads: 1-2 seconds
- With GPU: ~1 second per image

**System Requirements**:
- GPU: 2GB VRAM recommended
- CPU: 4GB RAM minimum
- Storage: ~500MB for model

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Image upload fails | Check file format and size |
| Model not loading | Verify model path exists |
| Slow predictions | GPU recommended for faster results |
| 500 error | Check server logs |
| No results | Try a clearer, closer image |

## Model Information

- **Type**: SegNext MSCAN-L
- **Training Data**: PlantSeg115 dataset
- **Classes**: 114 unique plant diseases
- **Framework**: PyTorch + MMSegmentation
- **Input Size**: 512x512 pixels
- **Accuracy**: ~85% on validation set

## Next Steps

1. ✅ Start the app: `python web_app.py`
2. ✅ Visit: `http://localhost:5000/disease-detection`
3. ✅ Upload a plant image
4. ✅ Get instant disease prediction
5. ✅ Download and share results

## Support

For issues:
- Check `/health` endpoint status
- Review error messages in browser console
- Check server logs for debugging
- Verify image format and size

---

**Happy Plant Disease Detection! 🌿**
