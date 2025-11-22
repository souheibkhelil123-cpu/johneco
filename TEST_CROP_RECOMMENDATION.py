"""
Test script for Crop Recommendation API endpoints
Run after starting web_app.py with: python web_app.py
"""

import json

# Test payloads and expected behavior

test_cases = [
    {
        "name": "MODE A: Get Optimal Conditions for Rice",
        "endpoint": "/api/get-plant-conditions",
        "method": "POST",
        "payload": {"crop_name": "rice"},
        "expected_fields": ["success", "crop_name", "optimal_conditions"],
        "curl": 'curl -X POST http://localhost:5000/api/get-plant-conditions -H "Content-Type: application/json" -d \'{"crop_name": "rice"}\''
    },
    {
        "name": "MODE B: Recommend Best Crop from Parameters",
        "endpoint": "/api/recommend-crop",
        "method": "POST",
        "payload": {
            "N": 90, "P": 42, "K": 43,
            "temperature": 23, "humidity": 75,
            "ph": 6.5, "rainfall": 200
        },
        "expected_fields": ["success", "recommended_crop", "confidence", "top_5_crops"],
        "curl": 'curl -X POST http://localhost:5000/api/recommend-crop -H "Content-Type: application/json" -d \'{"N": 90, "P": 42, "K": 43, "temperature": 23, "humidity": 75, "ph": 6.5, "rainfall": 200}\''
    },
    {
        "name": "MODE C: Analyze Crop Suitability",
        "endpoint": "/api/analyze-crop-suitability",
        "method": "POST",
        "payload": {
            "crop_name": "rice",
            "N": 70, "P": 30, "K": 35,
            "temperature": 25, "humidity": 60,
            "ph": 7.5, "rainfall": 150
        },
        "expected_fields": ["success", "crop_name", "suitability_score", "suitability_level", "nutrient_status", "recommendations"],
        "curl": 'curl -X POST http://localhost:5000/api/analyze-crop-suitability -H "Content-Type: application/json" -d \'{"crop_name": "rice", "N": 70, "P": 30, "K": 35, "temperature": 25, "humidity": 60, "ph": 7.5, "rainfall": 150}\''
    }
]

print("=" * 80)
print("🌾 CROP RECOMMENDATION API - TEST CASES")
print("=" * 80)
print("\nUse these curl commands or the web UI at http://localhost:5000/crop-recommendation\n")

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}️⃣  {test['name']}")
    print(f"   Endpoint: {test['endpoint']}")
    print(f"   Method: {test['method']}")
    print(f"   Expected response fields: {', '.join(test['expected_fields'])}")
    print(f"\n   cURL command:")
    print(f"   {test['curl']}")
    print(f"\n   Payload: {json.dumps(test['payload'], indent=6)}")
    print("-" * 80)

print("\n" + "=" * 80)
print("WEB UI ACCESS:")
print("=" * 80)
print("1. Start the Flask server: python web_app.py")
print("2. Open browser: http://localhost:5000/crop-recommendation")
print("3. Use the three tabs to test different modes:")
print("   - Tab 1: Select crop → See optimal conditions")
print("   - Tab 2: Enter parameters → Get best crop recommendation")
print("   - Tab 3: Select crop + parameters → Get suitability analysis & recommendations")
print("=" * 80)

print("\n" + "=" * 80)
print("AVAILABLE CROPS (for all endpoints):")
print("=" * 80)
crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton',
    'sugarcane', 'tobacco'
]
for i, crop in enumerate(crops, 1):
    print(f"  {i:2d}. {crop}")
print("=" * 80)
