"""
Usage Examples for Recommendation AI System
"""

from recommendation_engine import get_recommendation_engine
from crop_recommender import get_crop_recommender
from fertilizer_recommender import get_fertilizer_recommender


def example_1_full_recommendation():
    """Example 1: Get full recommendation based on sensor data"""
    print("=" * 70)
    print("EXAMPLE 1: Full Recommendation Based on Sensor Data")
    print("=" * 70)
    
    engine = get_recommendation_engine()
    
    # Sensor readings from farm
    N = 90
    P = 42
    K = 43
    temperature = 22
    humidity = 65
    ph = 6.5
    rainfall = 150
    
    print(f"\nSensor Data:")
    print(f"  Nitrogen (N): {N}")
    print(f"  Phosphorus (P): {P}")
    print(f"  Potassium (K): {K}")
    print(f"  Temperature: {temperature}°C")
    print(f"  Humidity: {humidity}%")
    print(f"  pH: {ph}")
    print(f"  Rainfall: {rainfall}mm\n")
    
    recommendation = engine.get_full_recommendation(
        N, P, K, temperature, humidity, ph, rainfall
    )
    
    if recommendation['success']:
        crop_rec = recommendation['crop_recommendation']
        print(f"🌾 Recommended Crop: {crop_rec['recommended_crop']}")
        print(f"   Confidence: {crop_rec['confidence']:.2%}")
        print(f"\n   Top 5 Recommendations:")
        for i, crop in enumerate(crop_rec['top_5_crops'], 1):
            print(f"   {i}. {crop['crop']}: {crop['probability']:.2%}")
        
        fert_rec = recommendation['fertilizer_recommendation']
        print(f"\n🥗 Fertilizer Recommendation: {fert_rec['recommendation']}")
        print(f"   Reason: {fert_rec['reason']}")
        print(f"   Application Rate: {fert_rec['application_rate']}")
        print(f"   Instructions:")
        for instruction in fert_rec['instructions']:
            print(f"     • {instruction}")
        
        soil = recommendation['soil_analysis']
        if soil.get('recommendations'):
            print(f"\n📊 Soil Adjustments Needed:")
            for rec in soil['recommendations']:
                print(f"   • {rec}")
    else:
        print(f"Error: {recommendation['error']}")
    
    print("\n")


def example_2_crop_specific():
    """Example 2: Get recommendation for specific crop"""
    print("=" * 70)
    print("EXAMPLE 2: Specific Crop Analysis (Rice)")
    print("=" * 70)
    
    engine = get_recommendation_engine()
    
    # Test for Rice
    N = 75
    P = 40
    K = 35
    temperature = 23
    humidity = 80
    ph = 6.5
    rainfall = 200
    
    print(f"\nSensor Data for Rice:")
    print(f"  N: {N}, P: {P}, K: {K}")
    print(f"  Temp: {temperature}°C, Humidity: {humidity}%")
    print(f"  pH: {ph}, Rainfall: {rainfall}mm\n")
    
    recommendation = engine.get_crop_specific_recommendation(
        'rice', N, P, K, temperature, humidity, ph, rainfall
    )
    
    if recommendation['success']:
        print(f"🌾 Crop: {recommendation['crop']}")
        feasibility = recommendation['feasibility']
        print(f"   Feasibility Score: {feasibility['score']}/100")
        print(f"   Rating: {feasibility['rating']}")
        
        print(f"\n📋 Recommendation:")
        print(recommendation['overall_recommendation'])
    else:
        print(f"Error: {recommendation['error']}")
    
    print("\n")


def example_3_fertilizer_analysis():
    """Example 3: Fertilizer analysis"""
    print("=" * 70)
    print("EXAMPLE 3: Fertilizer Analysis for Maize with Deficiencies")
    print("=" * 70)
    
    fert_recommender = get_fertilizer_recommender()
    crop_recommender = get_crop_recommender()
    
    # Maize with nitrogen deficiency
    crop = 'maize'
    optimal = crop_recommender.get_optimal_conditions(crop)
    
    current_N = 60  # Below optimal 120
    current_P = 50  # Near optimal 50
    current_K = 40  # Optimal 40
    
    print(f"\nCrop: {crop}")
    print(f"Current Soil: N={current_N}, P={current_P}, K={current_K}")
    print(f"Optimal Soil: N={optimal['N']}, P={optimal['P']}, K={optimal['K']}\n")
    
    recommendation = fert_recommender.recommend(
        crop,
        current_N, current_P, current_K,
        optimal['N'], optimal['P'], optimal['K']
    )
    
    print(f"✓ Recommendation: {recommendation['recommendation']}")
    print(f"✓ Application Rate: {recommendation['application_rate']}")
    print(f"✓ Reason: {recommendation['reason']}\n")
    
    print("📋 Instructions:")
    for instruction in recommendation['instructions']:
        print(f"  • {instruction}")
    
    print("\n📊 Nutrient Status:")
    for nutrient, status in recommendation['nutrient_status'].items():
        print(f"  {nutrient}: {status['status']} (Current: {status['current']}, Optimal: {status['optimal']})")
    
    print("\n")


def example_4_crop_analysis():
    """Example 4: Crop suitability analysis"""
    print("=" * 70)
    print("EXAMPLE 4: Crop Suitability Analysis")
    print("=" * 70)
    
    crop_recommender = get_crop_recommender()
    
    crops_to_test = ['rice', 'maize', 'wheat']
    
    N, P, K, ph = 80, 45, 40, 6.5
    
    print(f"\nAnalyzing crops for soil: N={N}, P={P}, K={K}, pH={ph}\n")
    
    for crop in crops_to_test:
        analysis = crop_recommender.get_analysis(crop, N, P, K, ph)
        
        print(f"🌾 {crop.upper()}")
        print(f"   Optimal: N={analysis['optimal_conditions']['N']}, "
              f"P={analysis['optimal_conditions']['P']}, "
              f"K={analysis['optimal_conditions']['K']}")
        print(f"   Current: N={analysis['current_conditions']['N']}, "
              f"P={analysis['current_conditions']['P']}, "
              f"K={analysis['current_conditions']['K']}")
        
        print(f"   Nutrient Status:")
        for nutrient, status in analysis['nutrient_status'].items():
            symbol = "✓" if status == "Optimal" else "⚠" if "deficiency" in status else "✗"
            print(f"     {symbol} {nutrient}: {status}")
        
        if analysis['recommendations']:
            print(f"   Recommendations:")
            for rec in analysis['recommendations']:
                print(f"     • {rec}")
        print()
    
    print("\n")


def example_5_all_crops():
    """Example 5: List all available crops"""
    print("=" * 70)
    print("EXAMPLE 5: Available Crops")
    print("=" * 70)
    
    crop_recommender = get_crop_recommender()
    
    print(f"\nTotal crops available: {len(crop_recommender.crops)}\n")
    
    # Organize by category
    cereals = ['rice', 'maize', 'wheat']
    pulses = ['chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil']
    fruits = ['banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya']
    others = ['pomegranate', 'coconut', 'cotton', 'sugarcane', 'tobacco']
    
    print("🌾 CEREALS:")
    for crop in cereals:
        print(f"  • {crop}")
    
    print("\n🫘 PULSES:")
    for crop in pulses:
        print(f"  • {crop}")
    
    print("\n🍌 FRUITS:")
    for crop in fruits:
        print(f"  • {crop}")
    
    print("\n🌳 OTHERS:")
    for crop in others:
        print(f"  • {crop}")
    
    print("\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("RECOMMENDATION AI SYSTEM - USAGE EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_1_full_recommendation()
    example_2_crop_specific()
    example_3_fertilizer_analysis()
    example_4_crop_analysis()
    example_5_all_crops()
    
    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)
