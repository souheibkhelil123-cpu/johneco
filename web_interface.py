#!/usr/bin/env python
"""
Flask web server for plant disease prediction via image upload.
Connects to the trained PlantSeg segmentation model.
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

# Disease class mapping (PlantSeg115)
DISEASE_CLASSES = {
    0: "Apple: Apple Black Rot",
    1: "Apple: Apple Mosaic Virus",
    2: "Apple: Apple Rust",
    3: "Apple: Apple Scab",
    4: "Banana: Banana Anthracnose",
    5: "Banana: Banana Black Leaf Streak",
    6: "Banana: Banana Bunchy Top",
    7: "Banana: Banana Cigar End Rot",
    8: "Banana: Banana Cordana Leaf Spot",
    9: "Banana: Banana Panama Disease",
    10: "Basil: Basil Downy Mildew",
    11: "Bean: Bean Halo Blight",
    12: "Bean: Bean Mosaic Virus",
    13: "Bean: Bean Rust",
    14: "Bell Pepper: Bell Pepper Bacterial Spot",
    15: "Bell Pepper: Bell Pepper Blossom End Rot",
    16: "Bell Pepper: Bell Pepper Frogeye Leaf Spot",
    17: "Bell Pepper: Bell Pepper Powdery Mildew",
    18: "Blueberry: Blueberry Anthracnose",
    19: "Blueberry: Blueberry Botrytis Blight",
    20: "Blueberry: Blueberry Mummy Berry",
    21: "Blueberry: Blueberry Rust",
    22: "Blueberry: Blueberry Scorch",
    23: "Broccoli: Broccoli Alternaria Leaf Spot",
    24: "Broccoli: Broccoli Downy Mildew",
    25: "Broccoli: Broccoli Ring Spot",
    26: "Cabbage: Cabbage Alternaria Leaf Spot",
    27: "Cabbage: Cabbage Black Rot",
    28: "Cabbage: Cabbage Downy Mildew",
    29: "Carrot: Carrot Alternaria Leaf Blight",
    30: "Carrot: Carrot Cavity Spot",
    31: "Carrot: Carrot Cercospora Leaf Blight",
    32: "Cauliflower: Cauliflower Alternaria Leaf Spot",
    33: "Cauliflower: Cauliflower Bacterial Soft Rot",
    34: "Celery: Celery Anthracnose",
    35: "Celery: Celery Early Blight",
    36: "Cherry: Cherry Leaf Spot",
    37: "Cherry: Cherry Powdery Mildew",
    38: "Citrus: Citrus Canker",
    39: "Citrus: Citrus Greening Disease",
    40: "Coffee: Coffee Berry Blotch",
    41: "Coffee: Coffee Black Rot",
    42: "Coffee: Coffee Brown Eye Spot",
    43: "Coffee: Coffee Leaf Rust",
    44: "Corn: Corn Gray Leaf Spot",
    45: "Corn: Corn Northern Leaf Blight",
    46: "Corn: Corn Rust",
    47: "Corn: Corn Smut",
    48: "Cucumber: Cucumber Angular Leaf Spot",
    49: "Cucumber: Cucumber Bacterial Wilt",
    50: "Cucumber: Cucumber Powdery Mildew",
    51: "Eggplant: Eggplant Cercospora Leaf Spot",
    52: "Eggplant: Eggplant Phomopsis Fruit Rot",
    53: "Eggplant: Eggplant Phytophthora Blight",
    54: "Garlic: Garlic Leaf Blight",
    55: "Garlic: Garlic Rust",
    56: "Ginger: Ginger Leaf Spot",
    57: "Ginger: Ginger Sheath Blight",
    58: "Grape: Grape Black Rot",
    59: "Grape: Grape Downy Mildew",
    60: "Grape: Grape Leaf Spot",
    61: "Grapevine: Grapevine Leafroll Disease",
    62: "Lettuce: Lettuce Downy Mildew",
    63: "Lettuce: Lettuce Mosaic Virus",
    64: "Maple: Maple Tar Spot",
    65: "Peach: Peach Anthracnose",
    66: "Peach: Peach Brown Rot",
    67: "Peach: Peach Leaf Curl",
    68: "Peach: Peach Rust",
    69: "Peach: Peach Scab",
    70: "Plum: Plum Bacterial Spot",
    71: "Plum: Plum Brown Rot",
    72: "Plum: Plum Pocket Disease",
    73: "Plum: Plum Pox Virus",
    74: "Plum: Plum Rust",
    75: "Potato: Potato Early Blight",
    76: "Potato: Potato Late Blight",
    77: "Raspberry: Raspberry Fire Blight",
    78: "Raspberry: Raspberry Gray Mold",
    79: "Raspberry: Raspberry Leaf Spot",
    80: "Raspberry: Raspberry Yellow Rust",
    81: "Rice: Rice Blast",
    82: "Rice: Rice Sheath Blight",
    83: "Soybean: Soybean Bacterial Blight",
    84: "Soybean: Soybean Brown Spot",
    85: "Soybean: Soybean Downy Mildew",
    86: "Soybean: Soybean Frog Eye Leaf Spot",
    87: "Soybean: Soybean Mosaic",
    89: "Squash: Squash Powdery Mildew",
    90: "Strawberry: Strawberry Anthracnose",
    91: "Strawberry: Strawberry Leaf Scorch",
    92: "Tobacco: Tobacco Blue Mold",
    93: "Tobacco: Tobacco Brown Spot",
    94: "Tobacco: Tobacco Frogeye Leaf Spot",
    95: "Tobacco: Tobacco Mosaic Virus",
    96: "Tomato: Tomato Bacterial Leaf Spot",
    97: "Tomato: Tomato Early Blight",
    98: "Tomato: Tomato Late Blight",
    99: "Tomato: Tomato Leaf Mold",
    100: "Tomato: Tomato Mosaic Virus",
    101: "Tomato: Tomato Septoria Leaf Spot",
    102: "Tomato: Tomato Yellow Leaf Curl Virus",
    103: "Wheat: Wheat Bacterial Leaf Streak (Black Chaff)",
    104: "Wheat: Wheat Head Scab",
    105: "Wheat: Wheat Leaf Rust",
    106: "Wheat: Wheat Loose Smut",
    107: "Wheat: Wheat Powdery Mildew",
    108: "Wheat: Wheat Septoria Blotch",
    109: "Wheat: Wheat Stem Rust",
    110: "Wheat: Wheat Stripe Rust",
    111: "Zucchini: Zucchini Bacterial Wilt",
    112: "Zucchini: Zucchini Downy Mildew",
    113: "Zucchini: Zucchini Powdery Mildew",
    114: "Zucchini: Zucchini Yellow Mosaic Virus",
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
    Handles various image formats and sizes.
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
        
        # Resize to model input size (256x256)
        # Use LANCZOS for high-quality downsampling
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        
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
            config_path = 'PlantSeg/configs/segnext/segnext_simple_256.py'
            checkpoint_path = 'PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth'
            
            # Verify files exist
            if not os.path.exists(config_path):
                logger.error("Config not found: {}".format(config_path))
                return False
            
            if not os.path.exists(checkpoint_path):
                logger.error("Checkpoint not found: {}".format(checkpoint_path))
                return False
            
            logger.info("Loading model from {}...".format(checkpoint_path))
            model = init_model(config_path, checkpoint_path, device=device)
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
    Predict disease from image.
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
        
        # Get segmentation logits
        try:
            seg_logits = result.seg_logits.data.cpu().numpy()
            if seg_logits.size == 0:
                return None, 0, -1, "No segmentation output from model"
            
            # Get predictions: shape is (num_classes, height, width)
            pred_classes = np.argmax(seg_logits, axis=0)
            logger.info("Segmentation shape: {}, Prediction shape: {}".format(seg_logits.shape, pred_classes.shape))
        except Exception as e:
            logger.error("Error processing model output: {}".format(e))
            return None, 0, -1, "Error processing model output: {}".format(str(e))
        
        # For PlantSeg binary segmentation (background vs leaf):
        # The model was trained on binary masks: class 0=background, class 1=diseased leaf
        try:
            # Get max logit values per class
            max_logits_per_class = np.max(seg_logits, axis=(1, 2))
            
            # Get pixel-level predictions (argmax across classes)
            pred_classes = np.argmax(seg_logits, axis=0)
            
            # Count pixels classified as disease (class 1)
            total_pixels = pred_classes.size
            disease_pixels = np.sum(pred_classes == 1)
            disease_percentage = (100.0 * disease_pixels) / total_pixels if total_pixels > 0 else 0.0
            
            # Get logits for class 1 (disease/diseased leaf)
            class_1_logit = float(max_logits_per_class[1]) if len(max_logits_per_class) > 1 else -np.inf
            class_0_logit = float(max_logits_per_class[0])
            
            logger.info(f"Binary segmentation results:")
            logger.info(f"  Disease pixels: {disease_percentage:.1f}% of image")
            logger.info(f"  Class 0 (BG) logit: {class_0_logit:.4f}")
            logger.info(f"  Class 1 (Disease) logit: {class_1_logit:.4f}")
            
            # Decision threshold: if >10% of image is classified as disease
            # AND class 1 logit is reasonable (not too negative)
            disease_detected = (disease_percentage > 10.0) and (class_1_logit > -3.0)
            
            if disease_detected:
                # Model detected disease - use disease percentage as confidence
                disease_name = "Leaf Disease Detected"  
                confidence = min(95.0, max(30.0, 50.0 + (disease_percentage * 0.3)))
                predicted_class = 1
                logger.info(f"DISEASE DETECTED - {disease_percentage:.1f}% of image classified as diseased leaf")
            else:
                # No disease detected
                disease_name = "Healthy Plant / No Disease"
                confidence = min(95.0, max(30.0, 50.0 - (disease_percentage * 0.5)))
                predicted_class = 0
                logger.info(f"NO DISEASE - Only {disease_percentage:.1f}% detected as disease")
            
            confidence = np.clip(confidence, 20.0, 95.0)
            
            logger.info(f"FINAL: {disease_name} (Confidence: {confidence:.1f}%)")
            return disease_name, confidence, predicted_class, None
            
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
        'model_loaded': model is not None
    }), 200

if __name__ == '__main__':
    print("=" * 70)
    print("PLANT DISEASE DETECTOR - WEB SERVER (IMPROVED)")
    print("=" * 70)
    print("Device: {}".format(device))
    print("Allowed file types: {}".format(', '.join(ALLOWED_EXTENSIONS)))
    
    # Load model on startup
    if load_model():
        print("\n[*] Starting Flask server...")
        print("Visit http://localhost:5000 in your browser")
        print("\nFeatures:")
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
        print("  PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth")
