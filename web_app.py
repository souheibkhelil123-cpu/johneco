# ===============================================
# Agriculture AI Web App - Crop & Disease
# ===============================================
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from recommendation_ai.crop_recommender import CropRecommender

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===============================================
# INITIALIZE FLASK APP
# ===============================================
print(f"[INFO] Current working directory: {os.getcwd()}")
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
print(f"[INFO] Template folder (absolute): {template_dir}")
print(f"[INFO] Template folder exists: {os.path.exists(template_dir)}")
if os.path.exists(template_dir):
    print(f"[INFO] Files in template folder: {os.listdir(template_dir)}")
app = Flask(__name__, template_folder=template_dir, static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# ===============================================
# CROP RECOMMENDATION SETUP
# ===============================================
try:
    model_path = os.path.join(os.path.dirname(__file__), 'recommendation_ai', 'models', 'crop_recommender.pkl')
    crop_recommender = CropRecommender(model_path=model_path)
    print("[INFO] Crop Recommender loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load Crop Recommender: {e}")
    crop_recommender = None

# ===============================================
# DISEASE DETECTION SETUP
# ===============================================
disease_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {device}")

# Disease class mapping (PlantSeg115)
DISEASE_CLASSES = {
    0: "Healthy / Background",
    1: "Apple: Apple Mosaic Virus",
    2: "Apple: Apple Rust",
    3: "Apple: Apple Scab",
    4: "Apple: Apple Black Rot",
    5: "Banana: Banana Anthracnose",
    6: "Banana: Banana Black Leaf Streak",
    7: "Banana: Banana Bunchy Top",
    8: "Banana: Banana Cigar End Rot",
    9: "Banana: Banana Cordana Leaf Spot",
    10: "Banana: Banana Panama Disease",
    11: "Basil: Basil Downy Mildew",
    12: "Bean: Bean Halo Blight",
    13: "Bean: Bean Mosaic Virus",
    14: "Bean: Bean Rust",
    15: "Bell Pepper: Bell Pepper Bacterial Spot",
    16: "Bell Pepper: Bell Pepper Blossom End Rot",
    17: "Bell Pepper: Bell Pepper Frogeye Leaf Spot",
    18: "Bell Pepper: Bell Pepper Powdery Mildew",
    19: "Blueberry: Blueberry Anthracnose",
    20: "Blueberry: Blueberry Botrytis Blight",
    21: "Blueberry: Blueberry Mummy Berry",
    22: "Blueberry: Blueberry Rust",
    23: "Blueberry: Blueberry Scorch",
    24: "Broccoli: Broccoli Alternaria Leaf Spot",
    25: "Broccoli: Broccoli Downy Mildew",
    26: "Broccoli: Broccoli Ring Spot",
    27: "Cabbage: Cabbage Alternaria Leaf Spot",
    28: "Cabbage: Cabbage Black Rot",
    29: "Cabbage: Cabbage Downy Mildew",
    30: "Carrot: Carrot Alternaria Leaf Blight",
    31: "Carrot: Carrot Cavity Spot",
    32: "Carrot: Carrot Cercospora Leaf Blight",
    33: "Cauliflower: Cauliflower Alternaria Leaf Spot",
    34: "Cauliflower: Cauliflower Bacterial Soft Rot",
    35: "Celery: Celery Anthracnose",
    36: "Celery: Celery Early Blight",
    37: "Cherry: Cherry Leaf Spot",
    38: "Cherry: Cherry Powdery Mildew",
    39: "Citrus: Citrus Canker",
    40: "Citrus: Citrus Greening Disease",
    41: "Coffee: Coffee Berry Blotch",
    42: "Coffee: Coffee Black Rot",
    43: "Coffee: Coffee Brown Eye Spot",
    44: "Coffee: Coffee Leaf Rust",
    45: "Corn: Corn Gray Leaf Spot",
    46: "Corn: Corn Northern Leaf Blight",
    47: "Corn: Corn Rust",
    48: "Corn: Corn Smut",
    49: "Cucumber: Cucumber Angular Leaf Spot",
    50: "Cucumber: Cucumber Bacterial Wilt",
    51: "Cucumber: Cucumber Powdery Mildew",
    52: "Eggplant: Eggplant Cercospora Leaf Spot",
    53: "Eggplant: Eggplant Phomopsis Fruit Rot",
    54: "Eggplant: Eggplant Phytophthora Blight",
    55: "Garlic: Garlic Leaf Blight",
    56: "Garlic: Garlic Rust",
    57: "Ginger: Ginger Leaf Spot",
    58: "Ginger: Ginger Sheath Blight",
    59: "Grape: Grape Black Rot",
    60: "Grape: Grape Downy Mildew",
    61: "Grape: Grape Leaf Spot",
    62: "Grapevine: Grapevine Leafroll Disease",
    63: "Lettuce: Lettuce Downy Mildew",
    64: "Lettuce: Lettuce Mosaic Virus",
    65: "Maple: Maple Tar Spot",
    66: "Peach: Peach Anthracnose",
    67: "Peach: Peach Brown Rot",
    68: "Peach: Peach Leaf Curl",
    69: "Peach: Peach Rust",
    70: "Peach: Peach Scab",
    71: "Plum: Plum Bacterial Spot",
    72: "Plum: Plum Brown Rot",
    73: "Plum: Plum Pocket Disease",
    74: "Plum: Plum Pox Virus",
    75: "Plum: Plum Rust",
    76: "Potato: Potato Early Blight",
    77: "Potato: Potato Late Blight",
    78: "Raspberry: Raspberry Fire Blight",
    79: "Raspberry: Raspberry Gray Mold",
    80: "Raspberry: Raspberry Leaf Spot",
    81: "Raspberry: Raspberry Yellow Rust",
    82: "Rice: Rice Blast",
    83: "Rice: Rice Sheath Blight",
    84: "Soybean: Soybean Bacterial Blight",
    85: "Soybean: Soybean Brown Spot",
    86: "Soybean: Soybean Downy Mildew",
    87: "Soybean: Soybean Frog Eye Leaf Spot",
    88: "Soybean: Soybean Mosaic",
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
    115: "Unknown Disease"
}

# File upload config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_disease_model():
    """Load disease prediction model from finaleco folder"""
    global disease_model
    try:
        from mmseg.apis import init_model as mmseg_init_model
        
        model_path = r'D:\colabecothoughts\finaleco\PlantSeg\work_dirs\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512\iter_400.pth'
        config_path = r'D:\colabecothoughts\finaleco\PlantSeg\work_dirs\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512\segnext_mscan-l_1xb16-adamw-40k_plantseg115-512x512.py'
        
        if not Path(model_path).exists():
            print(f"[ERROR] Model file not found: {model_path}")
            return False
        
        if not Path(config_path).exists():
            print(f"[ERROR] Config file not found: {config_path}")
            return False
        
        print(f"[INFO] Loading disease model from finaleco...")
        disease_model = mmseg_init_model(config_path, model_path, device=device)
        print("[SUCCESS] Disease prediction model loaded!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load disease model: {str(e)}")
        return False

# ===============================================
# CROP RECOMMENDATION ENDPOINTS
# ===============================================

@app.route('/')
def index():
    """Main page with both features"""
    try:
        print(f"[INFO] Rendering index_unified.html from: {app.template_folder}")
        return render_template('index_unified.html')
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Failed to render template: {e}\nTraceback:\n{tb}")
        return f"<h1>Error loading template</h1><pre>{str(e)}\n{tb}</pre>", 500

@app.route('/test')
def test():
    """Test endpoint"""
    return {"status": "ok", "template_dir": template_dir, "exists": os.path.exists(template_dir)}

@app.route('/api/get-plant-conditions', methods=['POST'])
def get_plant_conditions():
    data = request.json
    crop_name = data.get('crop_name', '').lower().strip()
    
    if not crop_recommender:
        return jsonify({'success': False, 'error': 'Crop recommender not loaded'})
    
    optimal = crop_recommender.get_optimal_conditions(crop_name)
    if optimal:
        return jsonify({'success': True, 'optimal_conditions': optimal})
    else:
        return jsonify({'success': False, 'error': f'Crop "{crop_name}" not found'})

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    data = request.json
    N = float(data.get('N', 50))
    P = float(data.get('P', 50))
    K = float(data.get('K', 50))
    temperature = float(data.get('temperature', 25))
    humidity = float(data.get('humidity', 60))
    ph = float(data.get('ph', 6.5))
    rainfall = float(data.get('rainfall', 100))
    
    if not crop_recommender:
        return jsonify({'success': False, 'error': 'Crop recommender not loaded'})
    
    result = crop_recommender.recommend(N, P, K, temperature, humidity, ph, rainfall)
    if result['success']:
        # Convert top_5_crops to a flat list of crop names for frontend
        top_5 = result.get('top_5_crops', [])
        if top_5 and isinstance(top_5[0], dict):
            top_5 = [c['crop'] for c in top_5]
        return jsonify({
            'success': True,
            'recommended_crop': result['recommended_crop'],
            'confidence': result['confidence'],
            'confidence_percentage': round(result['confidence'] * 100, 2),
            'top_5_crops': top_5,
            'input_params': result['input_params']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})

@app.route('/api/analyze-crop-suitability', methods=['POST'])
def analyze_crop_suitability():
    data = request.json
    crop_name = data.get('crop_name', '').lower().strip()
    N = float(data.get('N', 50))
    P = float(data.get('P', 50))
    K = float(data.get('K', 50))
    temperature = float(data.get('temperature', 25))
    humidity = float(data.get('humidity', 60))
    ph = float(data.get('ph', 6.5))
    rainfall = float(data.get('rainfall', 100))

    if not crop_recommender:
        return jsonify({'success': False, 'error': 'Crop recommender not loaded'})

    optimal = crop_recommender.get_optimal_conditions(crop_name)
    if not optimal:
        return jsonify({'success': False, 'error': f'Crop "{crop_name}" not found'})

    score = 100
    summary = []
    
    if N < optimal['N'] * 0.7:
        score -= 15
        summary.append('Nitrogen is low.')
    elif N > optimal['N'] * 1.3:
        score -= 10
        summary.append('Nitrogen is high.')
    
    if P < optimal['P'] * 0.7:
        score -= 10
        summary.append('Phosphorus is low.')
    elif P > optimal['P'] * 1.3:
        score -= 7
        summary.append('Phosphorus is high.')
    
    if K < optimal['K'] * 0.7:
        score -= 10
        summary.append('Potassium is low.')
    elif K > optimal['K'] * 1.3:
        score -= 7
        summary.append('Potassium is high.')
    
    if ph < optimal['ph'] - 0.5:
        score -= 10
        summary.append('Soil is too acidic.')
    elif ph > optimal['ph'] + 0.5:
        score -= 10
        summary.append('Soil is too alkaline.')
    
    if abs(temperature - optimal['temperature']) > 5:
        score -= 8
        summary.append('Temperature is not ideal.')
    
    if abs(humidity - optimal['humidity']) > 20:
        score -= 8
        summary.append('Humidity is not ideal.')
    
    if abs(rainfall - optimal['rainfall']) > 50:
        score -= 7
        summary.append('Rainfall is not ideal.')

    score = max(0, min(100, score))
    suitability_level = (
        'Excellent' if score >= 85 else
        'Good' if score >= 70 else
        'Fair' if score >= 50 else
        'Poor'
    )

    rec_result = crop_recommender.recommend(N, P, K, temperature, humidity, ph, rainfall)
    best_crop = rec_result['recommended_crop'] if rec_result['success'] else None
    analysis = crop_recommender.get_analysis(crop_name, N, P, K, ph)

    return jsonify({
        'success': True,
        'suitability_score': score,
        'suitability_level': suitability_level,
        'summary': ' '.join(summary) if summary else 'All parameters are optimal.',
        'recommended_crop': best_crop,
        'nutrient_status': analysis.get('nutrient_status', {}),
        'recommendations': analysis.get('recommendations', []),
        'current_conditions': analysis.get('current_conditions', {}),
        'optimal_conditions': analysis.get('optimal_conditions', {})
    })

# ===============================================
# DISEASE DETECTION ENDPOINTS
# ===============================================

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    """Predict disease from uploaded image"""
    try:
        from mmseg.apis import inference_model
        
        if disease_model is None:
            return jsonify({
                'success': False,
                'error': 'Disease model not loaded. Please restart the application.'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        timestamp = str(int(np.random.random() * 1000000))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and predict
            image = Image.open(filepath)
            result = inference_model(disease_model, filepath)
            
            # Get segmentation
            seg_logits = result.seg_logits.data.cpu().numpy()
            pred_classes = np.argmax(seg_logits, axis=0)
            
            # Classify disease
            unique_classes, counts = np.unique(pred_classes, return_counts=True)
            max_idx = np.argmax(counts)
            disease_class = unique_classes[max_idx]
            confidence = (counts[max_idx] / pred_classes.size) * 100
            disease_name = DISEASE_CLASSES.get(disease_class, f"Unknown (Class {disease_class})")
            
            # Create visualization
            colors = {}
            for i in range(116):
                colors[i] = (
                    int((i * 73) % 256),
                    int((i * 127) % 256),
                    int((i * 31) % 256)
                )
            
            h, w = pred_classes.shape
            pred_colored = Image.new('RGB', (w, h))
            pixels = pred_colored.load()
            for y in range(h):
                for x in range(w):
                    class_idx = int(pred_classes[y, x])
                    pixels[x, y] = colors[class_idx]
            
            pred_colored_resized = pred_colored.resize(image.size, Image.NEAREST)
            
            # Save results (no text overlay)
            output_image = image.copy()
            result_basename = f"result_{timestamp}"
            predicted_path = os.path.join(RESULTS_FOLDER, f"{result_basename}_predicted.jpg")
            seg_path = os.path.join(RESULTS_FOLDER, f"{result_basename}_segmentation.png")
            output_image.save(predicted_path)
            pred_colored_resized.save(seg_path)
            
            # Cleanup
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'disease': disease_name,
                'disease_class': int(disease_class),
                'confidence': round(confidence, 2),
                'predicted_image': f"/results/{os.path.basename(predicted_path)}",
                'segmentation_map': f"/results/{os.path.basename(seg_path)}"
            })
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/results/<filename>')
def serve_results(filename):
    """Serve result images"""
    return send_from_directory(RESULTS_FOLDER, filename)

# ===============================================
# HEALTH & INFO ENDPOINTS
# ===============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'Online',
        'app': 'Agriculture AI - Crop & Disease Detection',
        'crop_recommendation': 'Ready' if crop_recommender else 'Not Available',
        'disease_detection': 'Ready' if disease_model else 'Loading...',
        'device': device
    })

# ===============================================
# STARTUP
# ===============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("AGRICULTURE AI WEB APPLICATION")
    print("="*70)
    
    print("\n[INFO] Loading models...")
    print("[1/2] Crop Recommendation Model...", end=" ")
    print("OK" if crop_recommender else "FAILED")
    
    print("[2/2] Disease Detection Model...", end=" ")
    model_ok = load_disease_model()
    print("OK" if model_ok else "FAILED (will retry on first use)")
    
    print("\n" + "="*70)
    print("APPLICATION READY")
    print("="*70)
    print("\nOpen your browser and go to:")
    print("  http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST /api/predict-disease (Disease detection)")
    print("  POST /api/recommend-crop (Crop recommendation)")
    print("  POST /api/get-plant-conditions")
    print("  POST /api/analyze-crop-suitability")
    print("  GET  /health (Status)")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
