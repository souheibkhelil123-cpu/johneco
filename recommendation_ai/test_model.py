"""
Test script for trained crop recommendation model
"""

from crop_recommender import CropRecommender
import json

# Load the trained model
print("=" * 60)
print("🌾 CROP RECOMMENDER MODEL TEST")
print("=" * 60)

rec = CropRecommender(model_path='models/crop_recommender.pkl')

# Test Case 1: Rice conditions
print("\n✓ Test 1: Rice-like conditions")
result1 = rec.recommend(N=90, P=42, K=43, temperature=23, humidity=75, ph=6.5, rainfall=200)
print(f"  Recommended: {result1['recommended_crop']}")
print(f"  Confidence: {result1['confidence']:.2%}")
print(f"  Top 3: {[c['crop'] for c in result1['top_5_crops'][:3]]}")

# Test Case 2: Maize conditions
print("\n✓ Test 2: Maize-like conditions")
result2 = rec.recommend(N=120, P=50, K=40, temperature=25, humidity=65, ph=6.5, rainfall=150)
print(f"  Recommended: {result2['recommended_crop']}")
print(f"  Confidence: {result2['confidence']:.2%}")
print(f"  Top 3: {[c['crop'] for c in result2['top_5_crops'][:3]]}")

# Test Case 3: Apple conditions
print("\n✓ Test 3: Apple-like conditions")
result3 = rec.recommend(N=80, P=40, K=60, temperature=15, humidity=60, ph=6.5, rainfall=100)
print(f"  Recommended: {result3['recommended_crop']}")
print(f"  Confidence: {result3['confidence']:.2%}")
print(f"  Top 3: {[c['crop'] for c in result3['top_5_crops'][:3]]}")

# Test Case 4: Get optimal conditions for a crop
print("\n✓ Test 4: Optimal conditions for Rice")
optimal = rec.get_optimal_conditions('rice')
print(f"  Optimal conditions: {json.dumps(optimal, indent=2)}")

# Test Case 5: Soil analysis
print("\n✓ Test 5: Soil analysis for Rice (current vs optimal)")
analysis = rec.get_analysis('rice', current_N=70, current_P=30, current_K=35, current_ph=7.5)
print(f"  Nitrogen: {analysis['nutrient_status']['N']}")
print(f"  Phosphorus: {analysis['nutrient_status']['P']}")
print(f"  Potassium: {analysis['nutrient_status']['K']}")
print(f"  pH: {analysis['nutrient_status']['ph']}")
print(f"  Recommendations:")
for rec_text in analysis['recommendations']:
    print(f"    - {rec_text}")

print("\n" + "=" * 60)
print("✅ All tests completed successfully!")
print("=" * 60)
