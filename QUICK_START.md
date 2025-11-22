# 🌿 Plant Disease Detector - Quick Start

Your AI-powered plant disease detection web interface is now running!

## ✅ Server Status
- **Status**: 🟢 RUNNING
- **Device**: GPU (CUDA)
- **Model**: Loaded successfully
- **Port**: 5000

## 🌐 Access the Web Interface

### Local Access (Your Computer)
```
http://localhost:5000
```

### Network Access (From Another Device)
```
http://10.13.0.25:5000
```
(Replace IP with your actual machine IP if different)

## 🚀 How to Use

1. **Open the Web Interface**
   - Go to http://localhost:5000 in your browser
   
2. **Upload a Plant Image**
   - Click the upload area
   - Or drag-and-drop an image
   - Supported formats: JPG, PNG, BMP
   
3. **Analyze**
   - Click the "🔍 Analyze Plant Disease" button
   - Wait 2-3 seconds for results
   
4. **View Results**
   - Disease name will be displayed
   - Confidence percentage shown
   - Try another image anytime

## 📊 Model Information

**Trained Model**: MSCAN-L Segmentation Network
- Classes: 114 plant diseases
- Input Size: 256x256 pixels
- Training: 1000 iterations on PlantSeg115 dataset
- Accuracy: Tested on uploaded images

## 🎯 Supported Plants & Diseases

**114 Total Disease Classes Including:**
- 🍎 Apple (4 diseases)
- 🍌 Banana (6 diseases)  
- 🍅 Tomato (7 diseases)
- 🌾 Wheat (7 diseases)
- 🥒 Cucumber, Bean, Corn, Pepper, Potato... and more!

## ⚙️ Server Commands

### Stop the Server
```bash
# Press Ctrl+C in the terminal
```

### Restart the Server
```bash
python web_interface.py
```

### Check Server Health
```bash
curl http://localhost:5000/health
```

## 📱 Tips for Best Results

✅ **Good Images**:
- Clear, well-lit photos
- Focus on the affected area
- Include the leaf/fruit with visible symptoms

❌ **Avoid**:
- Very blurry images
- Partial views
- Multiple plants in one image
- Low resolution photos

## 🔧 Configuration

To change port (if 5000 is busy):
1. Edit `web_interface.py`
2. Find: `app.run(debug=False, host='0.0.0.0', port=5000)`
3. Change to: `app.run(debug=False, host='0.0.0.0', port=8000)`
4. Restart server

## 🐛 Troubleshooting

**Can't access http://localhost:5000?**
- Check terminal shows "Running on..."
- Try http://127.0.0.1:5000 instead
- Disable browser cache (Ctrl+Shift+R)

**Slow predictions?**
- First prediction loads model (slower)
- Subsequent predictions are faster
- GPU provides 2-3x speedup vs CPU

**Out of memory error?**
- Close other applications
- Reduce batch size in code
- Check GPU memory: `nvidia-smi`

## 📚 API Usage (For Developers)

### Test with curl
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/api/predict
```

### Response Example
```json
{
    "success": true,
    "disease": "Tomato: Tomato Early Blight",
    "confidence": 92.45,
    "class_index": 97,
    "filename": "image.jpg"
}
```

## 📈 Next Steps

1. **Test the Interface**
   - Upload various plant images
   - Check accuracy of predictions
   
2. **Gather Feedback**
   - Note any misclassifications
   - Identify challenging cases
   
3. **Improve Model** (Optional)
   - Collect more training data
   - Retrain with: `bash run.sh` in PlantSeg folder
   
4. **Deploy** (Optional)
   - Use Gunicorn for production
   - Set up reverse proxy (Nginx)
   - Host on cloud platform

## 📝 Log Files

Server logs appear in terminal. For persistent logging:

```python
# In web_interface.py, add:
import logging
logging.basicConfig(filename='disease_detector.log', level=logging.INFO)
```

## 🎉 Success!

Your plant disease detector is ready to use!

For more information, see: **WEB_INTERFACE_README.md**

---
**Made with ❤️ using Flask, PyTorch, and MMSegmentation**
