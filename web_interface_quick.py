#!/usr/bin/env python
"""
Simple web interface for disease detection that works with quick checkpoints.
Works while training is in progress - automatically uses latest checkpoint.
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from PIL import Image
import os
from pathlib import Path
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disease class mapping (simplified)
DISEASE_CLASSES = {
    0: "Background/Healthy",
    1: "Apple: Black Rot",
    2: "Apple: Mosaic Virus",
    3: "Apple: Rust",
    4: "Apple: Scab",
    5: "Banana: Anthracnose",
    97: "Tomato: Early Blight",
    98: "Tomato: Late Blight",
}

model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_latest_checkpoint():
    """Find latest checkpoint from quick or full training"""
    # Check quick training first
    quick_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_quick')
    if quick_dir.exists():
        checkpoints = sorted(quick_dir.glob('iter_*.pth'))
        if checkpoints:
            return str(checkpoints[-1])
    
    # Check full training
    full_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    if full_dir.exists():
        checkpoints = sorted(full_dir.glob('iter_*.pth'))
        if checkpoints:
            return str(checkpoints[-1])
    
    return None

def load_model():
    """Load model from latest checkpoint"""
    global model
    if model is None:
        try:
            checkpoint_path = find_latest_checkpoint()
            
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.error("No checkpoint found")
                return False
            
            config_path = 'PlantSeg/configs/segnext/segnext_mscan-l_quick_plantseg115-512x512.py'
            
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            model = init_model(config_path, checkpoint_path, device=device)
            model.eval()
            torch.cuda.empty_cache()
            logger.info("[OK] Model loaded!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    return True

def predict_disease(image_path):
    """Run inference on image"""
    try:
        if model is None:
            return None, 0, -1, "Model not loaded"
        
        # Load and resize image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Run inference
        with torch.no_grad():
            result = inference_model(model, np.array(img_resized))
        
        # Get predictions
        seg_logits = result.seg_logits.data.cpu().numpy()
        pred_classes = np.argmax(seg_logits, axis=0)
        
        # Find most common class
        unique, counts = np.unique(pred_classes, return_counts=True)
        top_class = unique[np.argmax(counts)]
        
        disease_name = DISEASE_CLASSES.get(top_class, f"Class {top_class}")
        confidence = min(95, 50 + (np.max(counts) / pred_classes.size) * 50)
        
        logger.info(f"Predicted: {disease_name} ({confidence:.1f}%)")
        return disease_name, confidence, int(top_class), None
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0, -1, str(e)

@app.route('/')
def index():
    return render_template('disease_detector.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        import time
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        disease_name, confidence, class_idx, error = predict_disease(filepath)
        
        if error:
            os.remove(filepath)
            return jsonify({'error': error}), 500
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'class_index': class_idx
        }), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    checkpoint_path = find_latest_checkpoint()
    checkpoint_name = Path(checkpoint_path).name if checkpoint_path else "None"
    
    return jsonify({
        'status': 'running',
        'device': device,
        'model_loaded': model is not None,
        'latest_checkpoint': checkpoint_name
    }), 200

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTOR - WEB SERVER")
    print("="*70)
    print(f"Device: {device}")
    print("Loading model...\n")
    
    if load_model():
        print("\n[OK] Model loaded successfully!")
        print("[*] Starting web server at http://localhost:5000")
        print("[*] Upload plant images to detect diseases")
        print("[*] Press Ctrl+C to stop\n")
        
        try:
            app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
        except KeyboardInterrupt:
            print("\n[!] Server stopped")
    else:
        print("[ERROR] Failed to load model - training may still be in progress")
        print("Wait a few minutes for first checkpoint to be saved...")
        print("Checkpoint location: PlantSeg/work_dirs/segnext_mscan-l_quick/")
