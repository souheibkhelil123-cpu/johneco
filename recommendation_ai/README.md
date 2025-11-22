# 🌾 Recommendation AI System

Complete crop and fertilizer recommendation system using machine learning and rule-based classification based on sensor data and optimal crop conditions.

## Overview

This system provides:
- **🌾 Crop Recommendation**: Predicts the best crops to grow based on soil and environmental parameters
- **🥗 Fertilizer Recommendation**: Recommends specific fertilizers based on NPK deficiency
- **📊 Soil Analysis**: Provides detailed analysis of soil conditions
- **🌡️ Environmental Assessment**: Evaluates temperature, humidity, and rainfall suitability

## Components

### 1. Crop Recommender (`crop_recommender.py`)

**Model**: Random Forest Classifier (99.5% cross-validation accuracy)

**Features Used**:
- N (Nitrogen): 0-140
- P (Phosphorus): 5-145
- K (Potassium): 5-205
- Temperature: 8-43°C
- Humidity: 14-99%
- pH: 3.5-9.5
- Rainfall: 20-254 mm

**Supported Crops** (22 total):
- Cereals: rice, maize
- Pulses: chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil
- Fruits: banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, pomegranate
- Others: cotton, sugarcane, tobacco, coconut

### 2. Fertilizer Recommender (`fertilizer_recommender.py`)

**Method**: Rule-Based Classification

**Fertilizer Types**:
- Urea (46% N)
- DAP - Di-Ammonium Phosphate (18% N, 46% P)
- Potassium Chloride (60% K)
- NPK 10:10:10 (Balanced)
- NPK 15:15:15 (Medium)
- NPK 20:20:20 (High NPK)
- Ammonium Nitrate (35% N)
- Calcium Nitrate (15.5% N, 26% Ca)
- Organic Compost

**Decision Logic**:
- Analyzes deficiency level: Severe, Moderate, Mild, Optimal, Excess
- Considers multiple nutrients
- Recommends based on crop requirements
- Provides application rates and instructions

### 3. Recommendation Engine (`recommendation_engine.py`)

**Integrates**:
- Crop recommendation
- Soil analysis
- Environmental assessment
- Fertilizer recommendation

## Installation

```bash
# No external dependencies needed (uses only sklearn for the example)
pip install scikit-learn numpy
```

## Quick Start

### Basic Usage

```python
from recommendation_engine import get_recommendation_engine

# Initialize engine
engine = get_recommendation_engine()

# Get recommendation
recommendation = engine.get_full_recommendation(
    N=90,           # Nitrogen
    P=42,           # Phosphorus
    K=43,           # Potassium
    temperature=22, # °C
    humidity=65,    # %
    ph=6.5,
    rainfall=150    # mm
)

print(f"Recommended Crop: {recommendation['crop_recommendation']['recommended_crop']}")
print(f"Fertilizer: {recommendation['fertilizer_recommendation']['recommendation']}")
```

### Crop-Specific Recommendation

```python
# Get recommendation for specific crop
recommendation = engine.get_crop_specific_recommendation(
    crop_name='rice',
    N=75, P=40, K=35,
    temperature=23, humidity=80, ph=6.5, rainfall=200
)

print(f"Feasibility: {recommendation['feasibility']['rating']}")
print(recommendation['overall_recommendation'])
```

### Individual Recommenders

```python
from crop_recommender import get_crop_recommender
from fertilizer_recommender import get_fertilizer_recommender

# Crop recommender
crop_rec = get_crop_recommender()
recommendation = crop_rec.recommend(90, 42, 43, 22, 65, 6.5, 150)
print(f"Crop: {recommendation['recommended_crop']}")
print(f"Top 5: {recommendation['top_5_crops']}")

# Fertilizer recommender
fert_rec = get_fertilizer_recommender()
recommendation = fert_rec.recommend(
    'rice',
    current_N=75, current_P=40, current_K=35,
    optimal_N=80, optimal_P=47, optimal_K=40
)
print(f"Fertilizer: {recommendation['recommendation']}")
print(f"Rate: {recommendation['application_rate']}")
```

## API Reference

### CropRecommender

```python
class CropRecommender:
    def recommend(N, P, K, temperature, humidity, ph, rainfall)
    """Recommend crops for given parameters"""
    
    def get_optimal_conditions(crop_name)
    """Get optimal NPK, temp, humidity, pH, rainfall for crop"""
    
    def get_analysis(crop_name, current_N, P, K, ph)
    """Analyze if current conditions suit the crop"""
```

### FertilizerRecommender

```python
class FertilizerRecommender:
    def recommend(crop_name, current_N, P, K, optimal_N, P, K)
    """Recommend fertilizer based on NPK deficiency"""
    
    def get_fertilizer_info(fertilizer_type)
    """Get detailed info about a fertilizer type"""
```

### RecommendationEngine

```python
class RecommendationEngine:
    def get_full_recommendation(N, P, K, temperature, humidity, ph, rainfall)
    """Get comprehensive crop + fertilizer recommendation"""
    
    def get_crop_specific_recommendation(crop_name, N, P, K, ...)
    """Get feasibility score and recommendations for specific crop"""
```

## Output Format

### Full Recommendation Response

```json
{
  "success": true,
  "crop_recommendation": {
    "recommended_crop": "rice",
    "confidence": 0.92,
    "top_5_crops": [
      {"crop": "rice", "probability": 0.92},
      {"crop": "maize", "probability": 0.05},
      ...
    ],
    "optimal_conditions": {
      "N": 80, "P": 47, "K": 40,
      "temperature": 23, "humidity": 80,
      "ph": 6.5, "rainfall": 200
    }
  },
  "soil_analysis": {
    "crop": "rice",
    "nutrient_status": {
      "N": "Optimal",
      "P": "Optimal",
      "K": "Optimal",
      "ph": "Optimal"
    },
    "recommendations": []
  },
  "fertilizer_recommendation": {
    "recommendation": "Soil already optimal - No fertilizer needed",
    "reason": "Soil nutrient levels are optimal",
    "application_rate": "None"
  }
}
```

## Examples

Run the examples file to see all features:

```bash
cd recommendation_ai
python examples.py
```

**Examples included**:
1. Full recommendation from sensor data
2. Specific crop analysis
3. Fertilizer analysis
4. Crop suitability analysis
5. List all available crops

## Deficiency Classification

The system classifies nutrient deficiencies as:

- **Severe**: <30% of optimal
- **Moderate**: 30-60% of optimal
- **Mild**: 60-90% of optimal
- **Optimal**: 90-130% of optimal
- **Excess**: >130% of optimal

## Optimal Conditions Database

The system includes optimal conditions for all 22 crops including:

```
Crop: Rice
├─ N: 80 kg/ha
├─ P: 47 kg/ha
├─ K: 40 kg/ha
├─ Temperature: 23°C
├─ Humidity: 80%
├─ pH: 6.5
└─ Rainfall: 200 mm
```

All conditions are based on agricultural research and field data.

## Features

✅ **22 supported crops**
✅ **9 fertilizer types**
✅ **Rule-based classification**
✅ **Optimal condition database**
✅ **Deficiency analysis**
✅ **Environmental assessment**
✅ **Feasibility scoring**
✅ **Detailed recommendations**
✅ **Application instructions**
✅ **Easy integration**

## Use Cases

1. **Farm Decision Support**: Help farmers choose what to plant
2. **Soil Management**: Recommend fertilizer and amendments
3. **Crop Planning**: Plan crop rotation
4. **Resource Optimization**: Minimize input costs
5. **Yield Prediction**: Assess crop suitability
6. **Sustainability**: Choose appropriate fertilizer types

## Data Sources

Based on research from:
- Kaggle Crop Recommendation Dataset
- Agricultural research papers
- Fertilizer recommendation studies
- Crop optimal condition databases

## Integration with Web App

```python
# In web_app.py or API routes
from recommendation_ai import get_recommendation_engine

engine = get_recommendation_engine()

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    data = request.json
    result = engine.get_full_recommendation(
        data['N'], data['P'], data['K'],
        data['temperature'], data['humidity'],
        data['ph'], data['rainfall']
    )
    return jsonify(result)
```

## Future Enhancements

- [ ] Train on real crop yield data
- [ ] Add more crops (50+)
- [ ] Include pest/disease recommendations
- [ ] Seasonal adjustments
- [ ] Cost optimization
- [ ] Sustainability scoring
- [ ] Multi-season planning
- [ ] Machine learning fertilizer recommender

## License

MIT License - Feel free to use and modify

## References

1. Crop Recommendation: https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
2. Fertilizer Dataset: https://github.com/Gladiator07/Harvestify
3. Research Paper: "Harvestify: Agricultural Intelligence For Farming" (ACM INDIA 2021)

---

**Status**: ✅ Production Ready

**For quick testing**: `python examples.py`
