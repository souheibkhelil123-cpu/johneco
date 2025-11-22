# Disease Prediction Configuration

This file documents all configurable parameters for the disease prediction feature.

## Model Configuration

### Model Paths
Located in `web_app.py` function `load_disease_model()`:

```python
# Path to model checkpoint
model_path = r'D:\colabecothoughts\finaleco\PlantSeg\work_dirs\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512\iter_400.pth'

# Path to model config
config_path = r'D:\colabecothoughts\finaleco\PlantSeg\work_dirs\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py'
```

**Customization**: Change these paths if you want to use a different model checkpoint (e.g., iter_200.pth for faster but less accurate results).

## File Upload Configuration

### Upload Folder
```python
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
```
Temporary storage for uploaded images. Auto-deleted after processing.

### Results Folder
```python
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
```
Storage for prediction results and visualizations. Can be served directly.

### Allowed File Extensions
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
```

**Customization**: Add or remove image formats as needed:
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
```

### File Size Limit
```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

**Customization**: Adjust maximum upload size (in bytes):
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024   # 25MB
```

## Device Configuration

### GPU/CPU Selection
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Customization**: Force specific device:
```python
device = 'cuda'    # Force GPU
device = 'cpu'     # Force CPU
device = 'cuda:0'  # Use specific GPU
device = 'cuda:1'  # Use second GPU
```

## Disease Classes Configuration

### Disease Mapping
Located in `web_app.py`:

```python
DISEASE_CLASSES = {
    0: "Apple: Apple Black Rot",
    1: "Apple: Apple Mosaic Virus",
    # ... 114 total classes
}
```

**Customization**: Modify disease names:
- Classes 0-114 correspond to model output
- Class 0 is often background
- Update class names to match your language/requirements

Example modification:
```python
DISEASE_CLASSES = {
    0: "Healthy / Background",
    1: "苹果黑腐病",  # Chinese
    2: "Anthracnose de la Pomme",  # French
    # etc.
}
```

## Server Configuration

### Port
```python
app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
```

**Customization**: Change port:
```python
app.run(debug=False, host='0.0.0.0', port=8080, use_reloader=False)
```

### Host Binding
```python
host='0.0.0.0'  # Accessible from any IP
```

**Customization**:
```python
host='127.0.0.1'  # Localhost only
host='0.0.0.0'    # Any IP address
```

### Debug Mode
```python
debug=False  # Production mode
```

**Customization**:
```python
debug=True   # Development with auto-reload
debug=False  # Production (recommended)
```

## Inference Configuration

### Input Image Size
Defined in model config (fixed):
```
crop_size = (256, 256) or (512, 512)
```
To change: Modify the config file in the model directory.

### Preprocessing
Defined in model config:
```
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
bgr_to_rgb = True
```

### Confidence Calculation
```python
confidence = (max_count / seg_map.size) * 100
```

**Customization**: Modify calculation in `predict_disease_from_segmentation()`:
```python
# Current: Percentage of pixels with disease
# Alternative: Use highest class probability
```

## Visualization Configuration

### Color Mapping
```python
colors = {}
for i in range(116):
    colors[i] = (
        int((i * 73) % 256),
        int((i * 127) % 256),
        int((i * 31) % 256)
    )
```

**Customization**: Use custom color scheme:
```python
# Grayscale
colors[i] = (i * 2, i * 2, i * 2)

# Rainbow
import colorsys
colors[i] = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(i/116, 1, 1))
```

### Output Image Format
```python
output_image.save(predicted_image_path)  # JPG format
pred_colored_resized.save(seg_map_path)  # PNG format
```

**Customization**: Change format or quality:
```python
output_image.save(predicted_image_path, 'PNG', quality=95)
```

### Font Configuration
```python
font = ImageFont.truetype("arial.ttf", 40)
font_small = ImageFont.truetype("arial.ttf", 30)
```

**Customization**: Use different fonts:
```python
font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 40)
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40)
```

## API Response Configuration

### Response Format
Standard JSON format. Customizable fields in `/api/predict-disease`:

```python
return jsonify({
    'success': True,
    'disease': disease_name,
    'disease_class': int(disease_class),
    'confidence': round(confidence, 2),
    'predicted_image': f"/results/{os.path.basename(predicted_image_path)}",
    'segmentation_map': f"/results/{os.path.basename(seg_map_path)}"
})
```

**Customization**: Add additional fields:
```python
'model_version': 'SegNext_MSCAN_L_v1.0',
'processing_time': elapsed_time,
'image_size': image.size,
'timestamp': datetime.now().isoformat(),
```

## Logging Configuration

### Current Logging
Console output via `print()` statements.

**Customization**: Add file logging:
```python
import logging
logging.basicConfig(
    filename='disease_prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Disease prediction: {disease_name}")
```

## Performance Tuning

### Batch Processing
Currently processes one image at a time.

**For Production**: Add queuing:
```python
from celery import Celery
app.config['broker_url'] = 'redis://localhost:6379'
```

### Caching
Currently no caching. Options:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def predict_disease_cached(image_path):
    # ... prediction logic
```

### Memory Management
Currently loads full image in memory. For large batches:
```python
# Stream processing
def predict_disease_streaming(image_path):
    # Process in chunks
```

## Browser & Frontend Configuration

### File Upload UI
Located in `templates/disease_detection.html`:

```javascript
const ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'};
const MAX_FILE_SIZE = 50 * 1024 * 1024;  // 50MB
```

### Styling
CSS variables in HTML file:
```css
--primary-color: #667eea;
--secondary-color: #764ba2;
--success-color: #4CAF50;
--error-color: #f44336;
```

**Customization**: Change brand colors in the CSS:
```css
--primary-color: #FF6B6B;  /* Red */
--secondary-color: #FF8787;  /* Light Red */
```

## Model Alternative Configuration

### Using Different Model Checkpoint
Modify in `load_disease_model()`:
```python
model_path = r'path/to/iter_200.pth'  # Faster but less accurate
model_path = r'path/to/iter_600.pth'  # More accurate but slower
```

### Using Different Architecture
Replace in `load_disease_model()`:
```python
# Instead of: segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py
# Use: segnext_mscan-m_1xb16-adamw-40k_plantseg115-512x512.py
```

## Environment Variables (Optional)

Create `.env` file:
```
MODEL_PATH=D:\colabecothoughts\finaleco\PlantSeg\work_dirs\...
CONFIG_PATH=D:\colabecothoughts\finaleco\PlantSeg\work_dirs\...
UPLOAD_FOLDER=./uploads
RESULTS_FOLDER=./results
DEVICE=cuda
PORT=5000
```

Then load in `web_app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH')
```

## Security Configuration

### CORS (Cross-Origin Resource Sharing)
Add if needed:
```python
from flask_cors import CORS
CORS(app)
```

### Authentication
Add for production:
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Implement authentication
```

### Rate Limiting
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: 'global')

@app.route('/api/predict-disease', methods=['POST'])
@limiter.limit("60 per hour")
def predict_disease():
    # ... prediction logic
```

## Database Configuration (For Future)

If storing predictions:
```python
# SQLAlchemy setup
from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
```

## Monitoring & Metrics

### Health Check Endpoint
Currently at `GET /health`. Returns:
```json
{
    "status": "Online",
    "app": "Agriculture AI Web App",
    "crop_recommendation_ai": "Ready",
    "disease_prediction_ai": "Ready"
}
```

**Customization**: Add metrics:
```python
{
    "status": "Online",
    "predictions_total": 1234,
    "average_confidence": 0.825,
    "uptime_seconds": 3600,
    "gpu_memory_usage": 2048
}
```

## Testing Configuration

### Unit Tests
```python
import unittest

class TestDiseasePrediction(unittest.TestCase):
    def test_api_health(self):
        # Test health endpoint
    
    def test_disease_prediction(self):
        # Test prediction with sample image
```

### Load Testing
```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:5000/api/predict-disease

# Using wrk
wrk -t4 -c100 -d30s http://localhost:5000/api/predict-disease
```

## Reference

**Files Modified**:
- `web_app.py` - Main application
- `templates/disease_detection.html` - Web interface

**Configuration Locations**:
- Model paths: `web_app.py` line ~149
- Upload settings: `web_app.py` line ~175
- Disease classes: `web_app.py` line ~26
- Server settings: `web_app.py` line ~530

**Model Configuration Files**:
- Main config: `...segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py`

---

**Last Updated**: November 22, 2025
**Configuration Version**: 1.0
