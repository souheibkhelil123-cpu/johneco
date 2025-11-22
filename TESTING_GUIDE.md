# 🧪 Testing Guide - Plant Disease Detector

Complete testing guide for verifying your plant disease detection system.

## ✅ Pre-Flight Checks

### 1. Verify Installation
```bash
# Check Python version (should be 3.9+)
python --version

# Check key packages installed
python -c "import torch, flask, cv2, mmengine; print('✅ All packages OK')"

# Check model file exists
ls PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth

# Check disease mapping
python -c "from web_interface import DISEASE_CLASSES; print(f'✅ {len(DISEASE_CLASSES)} diseases loaded')"
```

### 2. Start Server
```bash
# Start the Flask server
python web_interface.py

# Expected output:
# ✅ Model loaded successfully!
# 🚀 Starting Flask server...
# Running on http://127.0.0.1:5000
```

## 🧪 Manual Testing

### Test 1: Health Check
```bash
curl http://localhost:5000/health

# Expected response:
# {"status":"running","device":"cuda","model_loaded":true}
```

### Test 2: Web Interface
```bash
# Open in browser:
# http://localhost:5000

# Expected:
# - Beautiful web page loads
# - Upload area visible
# - No errors in browser console
```

### Test 3: Single Image Upload (Bash)
```bash
# Test with your testimage.jpg
curl -X POST -F "image=@testimage.jpg" http://localhost:5000/api/predict

# Expected response (JSON):
# {
#     "success": true,
#     "disease": "Apple: Apple Black Rot",
#     "confidence": 85.5,
#     "class_index": 0,
#     "filename": "testimage.jpg"
# }
```

### Test 4: Invalid File Type
```bash
# Create a text file
echo "not an image" > test.txt

# Try to upload (should fail gracefully)
curl -X POST -F "image=@test.txt" http://localhost:5000/api/predict

# Expected: Error response (not crash)
```

### Test 5: No File Upload
```bash
# Try without file (should fail gracefully)
curl -X POST http://localhost:5000/api/predict

# Expected: Error message
```

## 🖥️ Browser Testing

### Test 1: Drag and Drop
1. Open http://localhost:5000
2. Drag testimage.jpg to upload area
3. Image preview should appear
4. "Analyze" button should become active

### Test 2: Click Upload
1. Click the upload area
2. Select image from file dialog
3. Preview appears
4. Click "Analyze Plant Disease"
5. Loading spinner shows
6. Results appear after 2-3 seconds

### Test 3: Responsive Design
1. Open http://localhost:5000
2. Test on different screen sizes:
   - Desktop (1920x1080)
   - Tablet (768x1024)
   - Mobile (375x667)
3. Layout should adapt properly
4. Text should be readable
5. Buttons should be touchable

### Test 4: Multiple Predictions
1. Upload first image → Get result
2. Click "Analyze Another Image"
3. Upload different image → Get result
4. Verify results are different (if images are different)

### Test 5: Error Handling
1. Try uploading a corrupted image
2. Try uploading a 1px×1px image
3. Try uploading a very large image (>16MB)
4. Each should show appropriate error message

## 🤖 Model Testing

### Test 1: Inference Speed
```python
import time
from web_interface import model, predict_disease

image_path = "testimage.jpg"

# Warm up
predict_disease(image_path)

# Time actual inference
start = time.time()
disease, conf, cls_idx = predict_disease(image_path)
elapsed = time.time() - start

print(f"Inference time: {elapsed:.2f} seconds")
# Expected: 2-3 seconds
```

### Test 2: Disease Mapping
```python
from web_interface import DISEASE_CLASSES

# Check we have all classes
print(f"Total classes: {len(DISEASE_CLASSES)}")  # Should be ~114
print(f"Has class 0: {0 in DISEASE_CLASSES}")    # Should be True
print(f"Has class 113: {113 in DISEASE_CLASSES}") # Should be True
print(f"Has class 115: {115 in DISEASE_CLASSES}") # Should be False

# Print sample classes
for i in range(5):
    print(f"  Class {i}: {DISEASE_CLASSES[i]}")
```

### Test 3: Device Detection
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

from web_interface import device
print(f"Using device: {device}")
```

## 📊 Performance Testing

### Test 1: Response Time
```bash
# Single request
time curl -X POST -F "image=@testimage.jpg" http://localhost:5000/api/predict

# Should complete in <5 seconds
```

### Test 2: Memory Usage
```bash
# Before starting server
free -h  # Linux/Mac
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows

# After loading model and running inference
# GPU memory: nvidia-smi
# RAM: Task Manager or System Monitor

# Check no memory leaks after multiple requests
```

### Test 3: Concurrent Requests
```bash
# Send 5 sequential requests
for i in {1..5}; do
    echo "Request $i:"
    curl -X POST -F "image=@testimage.jpg" http://localhost:5000/api/predict
done
```

### Test 4: Large Image Handling
```bash
# Test with large image (but under 16MB limit)
# Convert image to larger size while keeping <16MB
# Verify it processes correctly
```

## 🔍 Debug Testing

### Enable Debug Mode
```python
# Edit web_interface.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Check Logs
```bash
# Watch terminal output for:
# - Model loading messages
# - Inference completions
# - Error traces
# - Request logs
```

### Browser Console (F12)
Check for:
- JavaScript errors
- Failed API calls
- Console warnings
- Network requests

## 🚨 Stress Testing

### Test Many Rapid Requests
```bash
# Create loop of requests
for i in {1..20}; do
    curl -X POST -F "image=@testimage.jpg" \
         http://localhost:5000/api/predict &
done
wait
```

### Monitor Resources
```bash
# Linux: Watch GPU and CPU
watch -n 1 nvidia-smi
watch -n 1 'free -h'

# Windows: Task Manager
# Mac: Activity Monitor
```

## ✔️ Validation Checklist

### Installation
- [ ] Python 3.9+ installed
- [ ] All packages in requirements installed
- [ ] Model checkpoint exists
- [ ] Config file exists

### Server
- [ ] Server starts without errors
- [ ] Model loads successfully
- [ ] GPU detected correctly
- [ ] Port 5000 accessible

### Web Interface
- [ ] Opens in browser
- [ ] Upload area visible
- [ ] Drag-drop works
- [ ] File selection works

### API
- [ ] `/api/predict` endpoint works
- [ ] Returns JSON response
- [ ] Health check works
- [ ] Error handling works

### Model
- [ ] Inference completes
- [ ] Results are reasonable
- [ ] Confidence scores valid (0-100)
- [ ] Class indices correct (0-113)

### Performance
- [ ] Inference time acceptable (2-3s)
- [ ] Memory stable
- [ ] No memory leaks
- [ ] Responsive UI

## 📋 Test Cases

### Test Case 1: Normal Flow
1. Upload apple disease image
2. Click analyze
3. Verify "Apple:" in result
4. Confidence > 50%
5. ✅ Pass

### Test Case 2: Different Plant
1. Upload tomato disease image
2. Click analyze
3. Verify "Tomato:" in result
4. ✅ Pass

### Test Case 3: Multiple Images
1. Upload 5 different images
2. Verify all process correctly
3. Results vary appropriately
4. ✅ Pass

### Test Case 4: Error Case
1. Upload invalid file
2. Verify error message appears
3. Can still upload new image
4. ✅ Pass

## 🐛 Common Test Failures

### Issue: Port Already in Use
```
Error: Address already in use
Solution: Kill existing process or use different port
```

### Issue: Model Not Found
```
Error: iter_1000.pth not found
Solution: Train model - cd PlantSeg && bash run.sh
```

### Issue: CUDA Out of Memory
```
Error: CUDA out of memory
Solution: 
  1. Close other applications
  2. Use device = 'cpu'
  3. Restart Python
```

### Issue: Slow Inference
```
Expected: 2-3 seconds
Check:
  1. First request loads model (~3s additional)
  2. GPU loaded? Check nvidia-smi
  3. GPU in use? Verify in task manager
```

## 📈 Performance Benchmarks

### Expected Metrics
| Metric | Expected | Acceptable |
|--------|----------|-----------|
| Server Start | 10s | <30s |
| Model Load | 5-10s | <20s |
| First Inference | 2-3s | <5s |
| Subsequent | 2-3s | <5s |
| GPU Memory | 1.2GB | <2GB |
| RAM Peak | 3GB | <6GB |

## ✅ Final Sign-Off

Once all tests pass:
- [ ] Server running stable
- [ ] API responding correctly
- [ ] UI functional and responsive
- [ ] Model predictions accurate
- [ ] No errors in logs
- [ ] Documentation complete

**Status**: ✅ Ready for Production

---

**Testing Completed**: [Date/Time]
**Tested By**: [Your Name]
**Test Environment**: [Your System]
**Result**: ✅ PASSED / ❌ FAILED

## Notes
```
[Add any additional notes or observations]
```
