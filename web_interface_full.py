#!/usr/bin/env python
"""
Flask web server for plant disease detection via SegNext full model.
Detects all 115+ plant disease classes from the PlantSeg dataset.
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
import traceback
import logging
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disease class mapping (PlantSeg115 - 115 disease classes)
DISEASE_CLASSES = {
    0: "Background/Healthy Plant",
    1: "Apple: Black Rot",
    2: "Apple: Mosaic Virus",
    3: "Apple: Rust",
    4: "Apple: Scab",
    5: "Banana: Anthracnose",
    6: "Banana: Black Leaf Streak",
    7: "Banana: Bunchy Top",
    8: "Banana: Cigar End Rot",
    9: "Banana: Cordana Leaf Spot",
    10: "Banana: Panama Disease",
    11: "Basil: Downy Mildew",
    12: "Bean: Halo Blight",
    13: "Bean: Mosaic Virus",
    14: "Bean: Rust",
    15: "Bell Pepper: Bacterial Spot",
    16: "Bell Pepper: Blossom End Rot",
    17: "Bell Pepper: Frogeye Leaf Spot",
    18: "Bell Pepper: Powdery Mildew",
    19: "Blueberry: Anthracnose",
    20: "Blueberry: Botrytis Blight",
    21: "Blueberry: Mummy Berry",
    22: "Blueberry: Rust",
    23: "Blueberry: Scorch",
    24: "Broccoli: Alternaria Leaf Spot",
    25: "Broccoli: Downy Mildew",
    26: "Broccoli: Ring Spot",
    27: "Cabbage: Alternaria Leaf Spot",
    28: "Cabbage: Black Rot",
    29: "Cabbage: Downy Mildew",
    30: "Carrot: Alternaria Leaf Blight",
    31: "Carrot: Cavity Spot",
    32: "Carrot: Cercospora Leaf Blight",
    33: "Cauliflower: Alternaria Leaf Spot",
    34: "Cauliflower: Bacterial Soft Rot",
    35: "Celery: Anthracnose",
    36: "Celery: Early Blight",
    37: "Cherry: Leaf Spot",
    38: "Cherry: Powdery Mildew",
    39: "Citrus: Canker",
    40: "Citrus: Greening Disease",
    41: "Coffee: Berry Blotch",
    42: "Coffee: Black Rot",
    43: "Coffee: Brown Eye Spot",
    44: "Coffee: Leaf Rust",
    45: "Corn: Gray Leaf Spot",
    46: "Corn: Northern Leaf Blight",
    47: "Corn: Rust",
    48: "Corn: Smut",
    49: "Cucumber: Angular Leaf Spot",
    50: "Cucumber: Bacterial Wilt",
    51: "Cucumber: Powdery Mildew",
    52: "Eggplant: Cercospora Leaf Spot",
    53: "Eggplant: Phomopsis Fruit Rot",
    54: "Eggplant: Phytophthora Blight",
    55: "Garlic: Leaf Blight",
    56: "Garlic: Rust",
    57: "Ginger: Leaf Spot",
    58: "Ginger: Sheath Blight",
    59: "Grape: Black Rot",
    60: "Grape: Downy Mildew",
    61: "Grape: Leaf Spot",
    62: "Grapevine: Leafroll Disease",
    63: "Lettuce: Downy Mildew",
    64: "Lettuce: Mosaic Virus",
    65: "Maple: Tar Spot",
    66: "Peach: Anthracnose",
    67: "Peach: Brown Rot",
    68: "Peach: Leaf Curl",
    69: "Peach: Rust",
    70: "Peach: Scab",
    71: "Plum: Bacterial Spot",
    72: "Plum: Brown Rot",
    73: "Plum: Pocket Disease",
    74: "Plum: Pox Virus",
    75: "Plum: Rust",
    76: "Potato: Early Blight",
    77: "Potato: Late Blight",
    78: "Raspberry: Fire Blight",
    79: "Raspberry: Gray Mold",
    80: "Raspberry: Leaf Spot",
    81: "Raspberry: Yellow Rust",
    82: "Rice: Blast",
    83: "Rice: Sheath Blight",
    84: "Soybean: Bacterial Blight",
    85: "Soybean: Brown Spot",
    86: "Soybean: Downy Mildew",
    87: "Soybean: Frog Eye Leaf Spot",
    88: "Soybean: Mosaic",
    89: "Squash: Powdery Mildew",
    90: "Strawberry: Anthracnose",
    91: "Strawberry: Leaf Scorch",
    92: "Tobacco: Blue Mold",
    93: "Tobacco: Brown Spot",
    94: "Tobacco: Frogeye Leaf Spot",
    95: "Tobacco: Mosaic Virus",
    96: "Tomato: Bacterial Leaf Spot",
    97: "Tomato: Early Blight",
    98: "Tomato: Late Blight",
    99: "Tomato: Leaf Mold",
    100: "Tomato: Mosaic Virus",
    101: "Tomato: Septoria Leaf Spot",
    102: "Tomato: Yellow Leaf Curl Virus",
    103: "Wheat: Bacterial Leaf Streak (Black Chaff)",
    104: "Wheat: Head Scab",
    105: "Wheat: Leaf Rust",
    106: "Wheat: Loose Smut",
    107: "Wheat: Powdery Mildew",
    108: "Wheat: Septoria Blotch",
    109: "Wheat: Stem Rust",
    110: "Wheat: Stripe Rust",
    111: "Zucchini: Bacterial Wilt",
    112: "Zucchini: Downy Mildew",
    113: "Zucchini: Powdery Mildew",
    114: "Zucchini: Yellow Mosaic Virus",
}

# Global model variable (loaded once)
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(filepath):
    """
    Validate image file format and readability.
    Returns: (is_valid, error_message)
    """
    try:
        # Try to open with PIL
        img = Image.open(filepath)
        img.verify()
        
        # Re-open after verify (verify closes the file)
        img = Image.open(filepath)
        
        # Check mode and convert if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
        
        # Check minimum size
        if img.size[0] < 50 or img.size[1] < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        # Check maximum size (prevent memory issues)
        if img.size[0] > 4096 or img.size[1] > 4096:
            return False, "Image too large (maximum 4096x4096 pixels)"
        
        return True, None
        
    except Image.UnidentifiedImageError:
        return False, "Invalid image format or corrupted file"
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

def preprocess_image(filepath):
    """
    Preprocess image for model inference.
    Model expects 512x512 images.
    Returns: (preprocessed_filepath, error_message)
    """
    try:
        # Open image
        img = Image.open(filepath)
        
        # Convert to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
        
        # Ensure RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        
        # Resize to model input size (512x512)
        # Use LANCZOS for high-quality downsampling
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save preprocessed image
        output_path = filepath.replace('.', '_processed.')
        img_resized.save(output_path, quality=95)
        
        logger.info("Image preprocessed: {} -> {}".format(filepath, output_path))
        return output_path, None
        
    except Exception as e:
        logger.error("Preprocessing error: {}".format(e))
        return None, "Failed to preprocess image: {}".format(str(e))

def load_model():
    """Load the segmentation model with error handling"""
    global model
    if model is None:
        try:
            # Try to find latest checkpoint
            work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
            
            if not work_dir.exists():
                logger.error("Work directory not found: {}".format(work_dir))
                return False
            
            # Find latest checkpoint
            checkpoints = sorted(work_dir.glob('iter_*.pth'))
            if not checkpoints:
                # Try best.pth
                checkpoint_path = work_dir / 'best.pth'
                if not checkpoint_path.exists():
                    logger.error("No checkpoints found in {}".format(work_dir))
                    return False
            else:
                checkpoint_path = checkpoints[-1]  # Get latest
            
            config_path = 'PlantSeg/configs/segnext/segnext_mscan-l_full_plantseg115-512x512.py'
            
            # Verify files exist
            if not os.path.exists(config_path):
                logger.error("Config not found: {}".format(config_path))
                return False
            
            if not os.path.exists(checkpoint_path):
                logger.error("Checkpoint not found: {}".format(checkpoint_path))
                return False
            
            logger.info("Loading model from {}...".format(checkpoint_path))
            model = init_model(config_path, str(checkpoint_path), device=device)
            model.eval()  # Set to evaluation mode
            torch.cuda.empty_cache()  # Clear GPU cache
            logger.info("[OK] Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error("Failed to load model: {}\n{}".format(e, traceback.format_exc()))
            return False
    return True

def predict_disease(image_path):
    """
    Predict disease from image using full SegNext model.
    Returns: (disease_name, confidence, class_index, error_message)
    """
    try:
        if model is None:
            return None, 0, -1, "Model not loaded"
        
        # Validate image first
        is_valid, error_msg = validate_image(image_path)
        if not is_valid:
            logger.error("Image validation failed: {}".format(error_msg))
            return None, 0, -1, error_msg
        
        # Preprocess image
        processed_path, preprocess_error = preprocess_image(image_path)
        if preprocess_error:
            logger.error("Image preprocessing failed: {}".format(preprocess_error))
            return None, 0, -1, preprocess_error
        
        # Run inference with error handling (no gradients needed)
        try:
            logger.info("Running inference on: {}".format(processed_path))
            with torch.no_grad():
                result = inference_model(model, processed_path)
        except Exception as e:
            logger.error("Inference failed: {}".format(e))
            return None, 0, -1, "Model inference failed: {}".format(str(e))
        
        # Get segmentation logits and predictions
        try:
            seg_logits = result.seg_logits.data.cpu().numpy()
            if seg_logits.size == 0:
                return None, 0, -1, "No segmentation output from model"
            
            # Get pixel-wise predictions: shape is (num_classes, height, width)
            pred_classes = np.argmax(seg_logits, axis=0)
            
            logger.info("Segmentation shape: {}, Predictions shape: {}".format(seg_logits.shape, pred_classes.shape))
            
            # Count class occurrences
            class_counts = Counter(pred_classes.flatten())
            
            # Get top predicted disease classes
            total_pixels = pred_classes.size
            
            # Sort by pixel count (descending)
            sorted_predictions = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top predictions:")
            for class_idx, pixel_count in sorted_predictions[:5]:
                class_name = DISEASE_CLASSES.get(class_idx, f"Unknown {class_idx}")
                percentage = (100.0 * pixel_count) / total_pixels
                logger.info(f"  Class {class_idx}: {class_name} - {percentage:.2f}% ({pixel_count} pixels)")
            
            # Get the top predicted class
            top_class_idx, top_pixels = sorted_predictions[0]
            top_percentage = (100.0 * top_pixels) / total_pixels
            
            # Get disease name
            disease_name = DISEASE_CLASSES.get(top_class_idx, f"Unknown Class {top_class_idx}")
            
            # Calculate confidence based on:
            # 1. Percentage of image predicted as this class
            # 2. Logit values for this class
            max_logits = np.max(seg_logits, axis=(1, 2))
            top_class_logit = float(max_logits[top_class_idx])
            
            # Confidence formula: scale by percentage and logit strength
            base_confidence = top_percentage
            logit_factor = min(1.0, max(0.0, (top_class_logit + 5.0) / 10.0))  # Normalize logit
            
            # If top class is background, look for best disease class
            if top_class_idx == 0:
                # Find best non-background class
                for class_idx, pixel_count in sorted_predictions:
                    if class_idx != 0:
                        disease_name = DISEASE_CLASSES.get(class_idx, f"Unknown Class {class_idx}")
                        disease_percentage = (100.0 * pixel_count) / total_pixels
                        disease_logit = float(max_logits[class_idx])
                        confidence = disease_percentage * (0.3 + 0.7 * logit_factor)
                        confidence = np.clip(confidence, 20.0, 95.0)
                        logger.info(f"Top disease (non-bg): {disease_name} - {confidence:.1f}%")
                        return disease_name, confidence, class_idx, None
                
                # No disease detected
                disease_name = "Healthy Plant / No Disease Detected"
                confidence = np.clip(95.0, 20.0, 95.0)
                logger.info(f"HEALTHY: {disease_name} - {confidence:.1f}%")
                return disease_name, confidence, 0, None
            else:
                # Disease detected
                confidence = base_confidence * (0.3 + 0.7 * logit_factor)
                confidence = np.clip(confidence, 20.0, 95.0)
                logger.info(f"DISEASE: {disease_name} - {confidence:.1f}%")
                return disease_name, confidence, top_class_idx, None
            
        except Exception as e:
            logger.error("Error analyzing predictions: {}".format(e))
            return None, 0, -1, "Error analyzing predictions: {}".format(str(e))
        
    except Exception as e:
        logger.error("Unexpected error in predict_disease: {}\n{}".format(e, traceback.format_exc()))
        return None, 0, -1, "Unexpected error: {}".format(str(e))

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('disease_detector.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for disease prediction with comprehensive error handling"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            logger.warning("Invalid file type: {}".format(file.filename))
            return jsonify({'error': 'Invalid file type. Allowed: {}'.format(", ".join(ALLOWED_EXTENSIONS))}), 400
        
        # Save uploaded file
        try:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            import time
            filename = "{}_{}".format(int(time.time()), filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info("File saved: {}".format(filepath))
        except Exception as e:
            logger.error("Error saving file: {}".format(e))
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Validate image
        is_valid, error_msg = validate_image(filepath)
        if not is_valid:
            logger.warning("Image validation failed: {}".format(error_msg))
            # Clean up
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': 'Image validation failed: {}'.format(error_msg)}), 400
        
        logger.info("Processing image: {}".format(filename))
        
        # Make prediction
        disease_name, confidence, class_idx, pred_error = predict_disease(filepath)
        
        if disease_name is None:
            logger.error("Prediction failed: {}".format(pred_error))
            # Clean up
            try:
                os.remove(filepath)
                processed_path = filepath.replace('.', '_processed.')
                if os.path.exists(processed_path):
                    os.remove(processed_path)
            except:
                pass
            return jsonify({'error': pred_error}), 500
        
        logger.info("[OK] Predicted: {} (Confidence: {:.2f}%)".format(disease_name, confidence))
        
        # Clean up uploaded files
        try:
            os.remove(filepath)
            processed_path = filepath.replace('.', '_processed.')
            if os.path.exists(processed_path):
                os.remove(processed_path)
        except:
            pass
        
        # Return result as JSON
        return jsonify({
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'class_index': class_idx,
            'filename': filename
        }), 200
        
    except Exception as e:
        logger.error("Unexpected error in api_predict: {}\n{}".format(e, traceback.format_exc()))
        return jsonify({'error': 'Unexpected server error: {}'.format(str(e))}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'device': device,
        'model_loaded': model is not None,
        'cuda_available': torch.cuda.is_available()
    }), 200

if __name__ == '__main__':
    print("=" * 70)
    print("PLANT DISEASE DETECTOR - WEB SERVER")
    print("SegNext Full PlantSeg115 (115 Disease Classes)")
    print("=" * 70)
    print("Device: {}".format(device))
    print("Allowed file types: {}".format(', '.join(ALLOWED_EXTENSIONS)))
    
    # Load model on startup
    if load_model():
        print("\n[*] Starting Flask server...")
        print("Visit http://localhost:5000 in your browser")
        print("\nFeatures:")
        print("  [OK] Predicts all 115 plant disease classes")
        print("  [OK] Full SegNext model with MSCAN-L backbone")
        print("  [OK] Supports: JPG, PNG, BMP, GIF, TIFF")
        print("  [OK] Max file size: 16MB")
        print("  [OK] Image size validation (50-4096 pixels)")
        print("  [OK] Automatic image preprocessing & resizing")
        print("  [OK] Comprehensive error handling")
        print("=" * 70)
        
        try:
            app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        except KeyboardInterrupt:
            print("\n\n[!] Server stopped by user")
        except Exception as e:
            logger.error("Server error: {}\n{}".format(e, traceback.format_exc()))
    else:
        print("[ERROR] Failed to load model. Cannot start server.")
        print("Make sure the checkpoint exists at:")
        print("  PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115/")
        print("\nTraining may still be in progress...")
