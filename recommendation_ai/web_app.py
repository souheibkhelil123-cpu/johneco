
# Crop Recommendation Web App (Flask)
import os
from flask import Flask, request, jsonify, render_template
from recommendation_ai.crop_recommender import CropRecommender

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'recommendation_ai', 'models', 'crop_recommender.pkl')
crop_recommender = CropRecommender(model_path=model_path)

app = Flask(__name__)

# ========== API ENDPOINTS FOR CROP RECOMMENDATION ==========

# Mode A: Get optimal conditions for a crop
@app.route('/api/get-plant-conditions', methods=['POST'])
def get_plant_conditions():
    data = request.json
    crop_name = data.get('crop_name', '').lower().strip()
    optimal = crop_recommender.get_optimal_conditions(crop_name)
    if optimal:
        return jsonify({'success': True, 'optimal_conditions': optimal})
    else:
        return jsonify({'success': False, 'error': f'Crop "{crop_name}" not found'})

# Mode B: Recommend best crop for input parameters
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
    result = crop_recommender.recommend(N, P, K, temperature, humidity, ph, rainfall)
    if result['success']:
        return jsonify({
            'success': True,
            'recommended_crop': result['recommended_crop'],
            'confidence': result['confidence'],
            'confidence_percentage': round(result['confidence'] * 100, 2),
            'top_5_crops': result['top_5_crops'],
            'input_params': result['input_params']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})

# Mode C: Analyze crop suitability and soil recommendations
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

    # Get optimal conditions
    optimal = crop_recommender.get_optimal_conditions(crop_name)
    if not optimal:
        return jsonify({'success': False, 'error': f'Crop "{crop_name}" not found'})

    # Suitability score calculation (simple heuristic)
    score = 100
    summary = []
    # N
    if N < optimal['N'] * 0.7:
        score -= 15
        summary.append('Nitrogen is low.')
    elif N > optimal['N'] * 1.3:
        score -= 10
        summary.append('Nitrogen is high.')
    # P
    if P < optimal['P'] * 0.7:
        score -= 10
        summary.append('Phosphorus is low.')
    elif P > optimal['P'] * 1.3:
        score -= 7
        summary.append('Phosphorus is high.')
    # K
    if K < optimal['K'] * 0.7:
        score -= 10
        summary.append('Potassium is low.')
    elif K > optimal['K'] * 1.3:
        score -= 7
        summary.append('Potassium is high.')
    # pH
    if ph < optimal['ph'] - 0.5:
        score -= 10
        summary.append('Soil is too acidic.')
    elif ph > optimal['ph'] + 0.5:
        score -= 10
        summary.append('Soil is too alkaline.')
    # Temperature
    if abs(temperature - optimal['temperature']) > 5:
        score -= 8
        summary.append('Temperature is not ideal.')
    # Humidity
    if abs(humidity - optimal['humidity']) > 20:
        score -= 8
        summary.append('Humidity is not ideal.')
    # Rainfall
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
    suitability_color = (
        'green' if score >= 85 else
        'yellowgreen' if score >= 70 else
        'orange' if score >= 50 else
        'red'
    )
    is_recommended = score >= 70

    # Get best recommended crop for these params
    rec_result = crop_recommender.recommend(N, P, K, temperature, humidity, ph, rainfall)
    best_crop = rec_result['recommended_crop'] if rec_result['success'] else None

    # Get nutrient status and recommendations
    analysis = crop_recommender.get_analysis(crop_name, N, P, K, ph)

    return jsonify({
        'success': True,
        'suitability_score': score,
        'suitability_level': suitability_level,
        'suitability_color': suitability_color,
        'summary': ' '.join(summary) if summary else 'All parameters are within optimal range.',
        'is_recommended': is_recommended,
        'recommended_crop': best_crop,
        'nutrient_status': analysis.get('nutrient_status', {}),
        'recommendations': analysis.get('recommendations', []),
        'current_conditions': analysis.get('current_conditions', {}),
        'optimal_conditions': analysis.get('optimal_conditions', {})
    })

# ========== PAGE ROUTE ========== 
@app.route('/crop-recommendation')
def crop_recommendation_page():
    return render_template('crop_recommendation.html')

# ========== HEALTH CHECK ========== 
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Online',
        'app': 'Crop Recommendation API',
        'crop_recommendation_ai': 'Ready'
    })

if __name__ == '__main__':
    print("🌱 Crop Recommendation API - Starting...")
    print("📊 Access the web app at: http://localhost:5000/crop-recommendation")
    print("Endpoints:")
    print("  /api/get-plant-conditions")
    print("  /api/recommend-crop")
    print("  /api/analyze-crop-suitability")
    print("  /health")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)

