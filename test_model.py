"""
PlantSeg Model Test Script
Simple standalone script to test the PlantSeg model with image uploads
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

# Add PlantSeg to path
PLANTSEG_PATH = Path(__file__).parent / "PlantSeg"
sys.path.insert(0, str(PLANTSEG_PATH))

# Import the model interface
from plantseg_inference import get_inferencer

app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global model
model = None
MODEL_STATUS = "Initializing..."

@app.before_request
def init_model():
    """Initialize model on first request"""
    global model, MODEL_STATUS
    if model is None:
        try:
            print("Loading PlantSeg model...")
            model = get_inferencer()
            if model.model is not None:
                MODEL_STATUS = "✓ Model Ready"
                print("✓ PlantSeg model loaded successfully!")
            else:
                MODEL_STATUS = "✗ Model Failed to Load"
                print("✗ Model failed to initialize")
        except Exception as e:
            MODEL_STATUS = f"✗ Error: {str(e)}"
            print(f"Error loading model: {str(e)}")

@app.route('/')
def index():
    """Main test page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/test-segment', methods=['POST'])
def test_segment():
    """Test segmentation with uploaded image"""
    global model, MODEL_STATUS
    
    try:
        # Check if model is ready
        if model is None or model.model is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized',
                'status': MODEL_STATUS
            }), 503
        
        # Check if image provided
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Load image from bytes
        print(f"Processing image: {file.filename}")
        image_data = file.read()
        image = model.load_image_from_bytes(image_data)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load image'
            }), 400
        
        print(f"Image loaded: {image.shape}")
        
        # Run segmentation
        print("Running segmentation...")
        seg_result = model.segment_image(image)
        
        if not seg_result.get('success', False):
            return jsonify({
                'success': False,
                'error': seg_result.get('error', 'Segmentation failed')
            }), 500
        
        print("Segmentation complete!")
        
        # Extract results
        seg_mask = seg_result['segmentation_mask'].squeeze()
        
        # Generate visualization
        vis_image = model.visualize_segmentation(image, seg_mask, alpha=0.6)
        vis_b64 = model.image_to_base64(vis_image)
        
        # Calculate statistics
        unique_classes = np.unique(seg_mask)
        class_pixels = {}
        total_pixels = seg_mask.size
        
        for class_id in unique_classes:
            pixel_count = np.sum(seg_mask == class_id)
            percentage = (pixel_count / total_pixels) * 100
            class_pixels[int(class_id)] = {
                'count': int(pixel_count),
                'percentage': round(percentage, 2)
            }
        
        # Class info
        class_info = {
            0: {'name': 'Background', 'color': 'black'},
            1: {'name': 'Plant/Leaves', 'color': 'green'},
            2: {'name': 'Stem', 'color': 'brown'},
            3: {'name': 'Roots', 'color': 'red'}
        }
        
        result = {
            'success': True,
            'filename': file.filename,
            'original_size': f"{image.shape[1]}x{image.shape[0]}",
            'model_input_size': "128x128",
            'visualization': f'data:image/png;base64,{vis_b64}',
            'classes': class_info,
            'statistics': class_pixels,
            'total_regions': len(unique_classes),
            'message': f'Segmentation successful! Detected {len(unique_classes)} regions.'
        }
        
        print(f"Result: {result['message']}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get current model status"""
    global model, MODEL_STATUS
    
    return jsonify({
        'status': MODEL_STATUS,
        'model_loaded': model is not None and model.model is not None,
        'model_type': 'DeepLabV3 + ResNet101',
        'input_size': [128, 128],
        'classes': 4,
        'class_names': ['Background', 'Plant/Leaves', 'Stem', 'Roots']
    })

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlantSeg Model Test</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #2d5016 0%, #4a7c20 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.05em;
            opacity: 0.9;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .section-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        
        .upload-box:hover {
            border-color: #764ba2;
            background: #f0f2ff;
            transform: translateY(-2px);
        }
        
        .upload-box.dragover {
            border-color: #764ba2;
            background: #e8ebff;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .upload-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .upload-box p {
            color: #666;
            font-size: 0.95em;
        }
        
        #fileInput {
            display: none;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-upload {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-upload:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-test {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-top: 20px;
        }
        
        .btn-test:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-test:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-clear {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-clear:hover {
            background: #e0e0e0;
        }
        
        .info-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            color: #004085;
            font-size: 0.9em;
        }
        
        .info-box strong {
            color: #003366;
        }
        
        .file-name {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            color: #333;
            font-size: 0.9em;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.95em;
            font-weight: 600;
        }
        
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        
        .status-error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-loading {
            background: #fff3cd;
            color: #856404;
        }
        
        .results-container {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #ddd;
        }
        
        .result-image {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
            max-height: 400px;
            object-fit: contain;
        }
        
        .result-stats {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #667eea;
            font-size: 0.9em;
        }
        
        .stat-label {
            font-weight: 600;
            color: #333;
        }
        
        .stat-value {
            color: #667eea;
            font-weight: 600;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #f5c6cb;
        }
        
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .class-breakdown {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        
        .class-card {
            background: white;
            padding: 12px;
            border-radius: 5px;
            border-left: 4px solid;
            font-size: 0.85em;
        }
        
        .class-card.bg-0 { border-left-color: #000; }
        .class-card.bg-1 { border-left-color: #00ff00; }
        .class-card.bg-2 { border-left-color: #8b4513; }
        .class-card.bg-3 { border-left-color: #ff0000; }
        
        .class-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .class-value {
            color: #667eea;
            font-weight: 600;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
            
            header h1 {
                font-size: 1.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧪 PlantSeg Model Test</h1>
            <p>Upload an image to test the segmentation model</p>
        </header>
        
        <div class="content">
            <!-- Left: Upload -->
            <div class="section">
                <h2 class="section-title">📤 Upload Image</h2>
                
                <div class="upload-box" id="uploadBox">
                    <div class="upload-icon">📸</div>
                    <h3>Drag & Drop or Click</h3>
                    <p>Drop your plant image here or click to select</p>
                    <p style="font-size: 0.85em; margin-top: 10px; color: #999;">
                        Formats: JPEG, PNG, BMP, TIFF
                    </p>
                </div>
                
                <input type="file" id="fileInput" accept="image/*">
                
                <div id="fileName" class="file-name" style="display: none;"></div>
                
                <div class="button-group">
                    <button class="btn-upload" onclick="selectFile()">Choose File</button>
                    <button class="btn-clear" onclick="clearFile()">Clear</button>
                </div>
                
                <button class="btn-test" id="testBtn" onclick="testSegmentation()" disabled>
                    Test Segmentation
                </button>
                
                <div id="modelStatus" class="status-indicator">
                    <span>🔄</span>
                    <span>Checking model...</span>
                </div>
                
                <div class="info-box">
                    <strong>ℹ️ Model Info:</strong><br>
                    Type: DeepLabV3 + ResNet101<br>
                    Classes: 4 (Background, Leaves, Stem, Roots)<br>
                    Input Size: 128×128 pixels
                </div>
            </div>
            
            <!-- Right: Results -->
            <div class="section">
                <h2 class="section-title">📊 Results</h2>
                
                <div id="resultsArea" style="display: none;">
                    <div id="resultContent"></div>
                </div>
                
                <div id="noResults" style="text-align: center; padding: 40px; color: #999;">
                    <p style="font-size: 1.2em; margin-bottom: 10px;">📭</p>
                    <p>Upload and test an image to see results here</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const testBtn = document.getElementById('testBtn');
        const resultsArea = document.getElementById('resultsArea');
        const noResults = document.getElementById('noResults');
        const resultContent = document.getElementById('resultContent');
        const modelStatus = document.getElementById('modelStatus');
        
        let selectedFile = null;
        
        // Initialize
        checkModelStatus();
        setupEventListeners();
        
        function setupEventListeners() {
            uploadBox.addEventListener('click', selectFile);
            uploadBox.addEventListener('dragover', handleDragOver);
            uploadBox.addEventListener('dragleave', handleDragLeave);
            uploadBox.addEventListener('drop', handleDrop);
            fileInput.addEventListener('change', handleFileSelect);
        }
        
        function selectFile() {
            fileInput.click();
        }
        
        function handleFileSelect(e) {
            const files = e.target.files;
            if (files.length > 0) {
                setFile(files[0]);
            }
        }
        
        function handleDragOver(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                setFile(files[0]);
            }
        }
        
        function setFile(file) {
            if (file.type.startsWith('image/')) {
                selectedFile = file;
                fileName.textContent = `📄 ${file.name}`;
                fileName.style.display = 'block';
                testBtn.disabled = false;
                uploadBox.style.opacity = '0.6';
            } else {
                alert('Please select a valid image file');
            }
        }
        
        function clearFile() {
            selectedFile = null;
            fileInput.value = '';
            fileName.style.display = 'none';
            testBtn.disabled = true;
            uploadBox.style.opacity = '1';
            resultsArea.style.display = 'none';
            noResults.style.display = 'block';
        }
        
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/model-status');
                const data = await response.json();
                
                let statusHtml = '';
                if (data.model_loaded) {
                    statusHtml = '<span>✅</span><span>Model Ready</span>';
                    modelStatus.className = 'status-indicator status-success';
                } else {
                    statusHtml = '<span>⏳</span><span>' + data.status + '</span>';
                    modelStatus.className = 'status-indicator status-loading';
                }
                modelStatus.innerHTML = statusHtml;
            } catch (error) {
                modelStatus.innerHTML = '<span>❌</span><span>Error checking status</span>';
                modelStatus.className = 'status-indicator status-error';
            }
        }
        
        async function testSegmentation() {
            if (!selectedFile) {
                alert('Please select an image first');
                return;
            }
            
            testBtn.disabled = true;
            resultsArea.innerHTML = '<div class="loading"><div class="spinner"></div><span>Processing image...</span></div>';
            resultsArea.style.display = 'block';
            noResults.style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/api/test-segment', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    displayError(data.error);
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                testBtn.disabled = false;
            }
        }
        
        function displayResults(data) {
            let html = `
                <div class="status-indicator status-success" style="margin-bottom: 15px;">
                    <span>✅</span>
                    <span>${data.message}</span>
                </div>
                
                <img src="${data.visualization}" class="result-image" alt="Segmentation result">
                
                <div class="result-stats">
                    <div class="stat-item">
                        <span class="stat-label">File</span>
                        <span class="stat-value">${data.filename}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Original Size</span>
                        <span class="stat-value">${data.original_size}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Model Input</span>
                        <span class="stat-value">${data.model_input_size}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Regions Detected</span>
                        <span class="stat-value">${data.total_regions}</span>
                    </div>
                </div>
                
                <div class="class-breakdown">
            `;
            
            // Add class breakdowns
            Object.entries(data.statistics).forEach(([classId, stats]) => {
                const classInfo = data.classes[classId];
                html += `
                    <div class="class-card bg-${classId}">
                        <div class="class-name">${classInfo.name}</div>
                        <div class="class-value">${stats.percentage}%</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                            ${stats.count.toLocaleString()} pixels
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            
            resultContent.innerHTML = html;
        }
        
        function displayError(error) {
            resultContent.innerHTML = `<div class="error-message"><strong>Error:</strong> ${error}</div>`;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 60)
    print("🧪 PlantSeg Model Test - Starting Server")
    print("=" * 60)
    print("\n📊 Model Status:")
    print("   Loading PlantSeg model on first request...")
    print("\n🌐 Web Interface:")
    print("   Open: http://localhost:5001")
    print("\n📝 Steps:")
    print("   1. Wait for model to load (watch console)")
    print("   2. Upload a plant image")
    print("   3. Click 'Test Segmentation'")
    print("   4. View results with overlay and statistics")
    print("\n💡 Tips:")
    print("   - Use clear, well-lit plant images")
    print("   - Full plant in frame works best")
    print("   - Check console for model loading status")
    print("\n" + "=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
