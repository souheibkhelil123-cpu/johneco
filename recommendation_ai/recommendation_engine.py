"""
Integrated Recommendation API
Combines crop and fertilizer recommendations
"""

from .crop_recommender import get_crop_recommender
from .fertilizer_recommender import get_fertilizer_recommender


class RecommendationEngine:
    """
    Integrated recommendation engine for crops and fertilizers
    """
    
    def __init__(self):
        """Initialize recommendation engines"""
        self.crop_recommender = get_crop_recommender()
        self.fertilizer_recommender = get_fertilizer_recommender()
    
    def get_full_recommendation(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Get comprehensive recommendation including crop and fertilizer suggestions
        
        Args:
            N: Nitrogen content (0-140)
            P: Phosphorus content (5-145)
            K: Potassium content (5-205)
            temperature: Temperature in Celsius (8-43)
            humidity: Relative humidity in % (14-99)
            ph: pH value of soil (3.5-9.5)
            rainfall: Rainfall in mm (20-254)
        
        Returns:
            Complete recommendation with crop and fertilizer
        """
        
        # Get crop recommendation
        crop_rec = self.crop_recommender.recommend(N, P, K, temperature, humidity, ph, rainfall)
        
        if not crop_rec['success']:
            return {
                'success': False,
                'error': crop_rec['error']
            }
        
        recommended_crop = crop_rec['recommended_crop']
        
        # Get optimal conditions for recommended crop
        optimal_conditions = self.crop_recommender.get_optimal_conditions(recommended_crop)
        
        # Get soil analysis for recommended crop
        soil_analysis = self.crop_recommender.get_analysis(
            recommended_crop, N, P, K, ph
        )
        
        # Get fertilizer recommendation
        fertilizer_rec = self.fertilizer_recommender.recommend(
            recommended_crop,
            N, P, K,
            optimal_conditions['N'],
            optimal_conditions['P'],
            optimal_conditions['K']
        )
        
        return {
            'success': True,
            'crop_recommendation': {
                'recommended_crop': recommended_crop,
                'confidence': crop_rec['confidence'],
                'top_5_crops': crop_rec['top_5_crops'],
                'optimal_conditions': optimal_conditions
            },
            'soil_analysis': soil_analysis,
            'fertilizer_recommendation': fertilizer_rec,
            'input_parameters': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        }
    
    def get_crop_specific_recommendation(self, crop_name, N, P, K, temperature, humidity, ph, rainfall):
        """
        Get recommendation for a specific crop chosen by user
        
        Args:
            crop_name: User-selected crop
            N, P, K: Soil nutrient values
            temperature, humidity, ph, rainfall: Environmental parameters
        
        Returns:
            Recommendation with feasibility analysis and fertilizer suggestion
        """
        
        # Get optimal conditions
        optimal_conditions = self.crop_recommender.get_optimal_conditions(crop_name)
        
        if not optimal_conditions:
            return {
                'success': False,
                'error': f'Crop {crop_name} not found in database'
            }
        
        # Analyze soil suitability
        soil_analysis = self.crop_recommender.get_analysis(crop_name, N, P, K, ph)
        
        # Check environmental suitability
        env_analysis = self._analyze_environment(
            crop_name, temperature, humidity, rainfall, optimal_conditions
        )
        
        # Get fertilizer recommendation
        fertilizer_rec = self.fertilizer_recommender.recommend(
            crop_name,
            N, P, K,
            optimal_conditions['N'],
            optimal_conditions['P'],
            optimal_conditions['K']
        )
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility(
            soil_analysis, env_analysis
        )
        
        return {
            'success': True,
            'crop': crop_name,
            'feasibility': {
                'score': feasibility_score,
                'rating': self._rate_feasibility(feasibility_score)
            },
            'soil_analysis': soil_analysis,
            'environmental_analysis': env_analysis,
            'fertilizer_recommendation': fertilizer_rec,
            'overall_recommendation': self._generate_recommendation_text(
                crop_name, soil_analysis, env_analysis, fertilizer_rec
            ),
            'input_parameters': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
        }
    
    def _analyze_environment(self, crop_name, temp, humidity, rainfall, optimal):
        """Analyze environmental conditions"""
        analysis = {
            'crop': crop_name,
            'current_conditions': {
                'temperature': temp,
                'humidity': humidity,
                'rainfall': rainfall
            },
            'optimal_conditions': {
                'temperature': optimal['temperature'],
                'humidity': optimal['humidity'],
                'rainfall': optimal['rainfall']
            },
            'issues': []
        }
        
        # Temperature analysis
        if temp < optimal['temperature'] - 5:
            analysis['issues'].append(f"⚠️ Temperature too low ({temp}°C vs optimal {optimal['temperature']}°C)")
        elif temp > optimal['temperature'] + 5:
            analysis['issues'].append(f"⚠️ Temperature too high ({temp}°C vs optimal {optimal['temperature']}°C)")
        
        # Humidity analysis
        if humidity < optimal['humidity'] - 10:
            analysis['issues'].append(f"⚠️ Humidity too low ({humidity}% vs optimal {optimal['humidity']}%)")
        elif humidity > optimal['humidity'] + 15:
            analysis['issues'].append(f"⚠️ Humidity too high ({humidity}% vs optimal {optimal['humidity']}%)")
        
        # Rainfall analysis
        if rainfall < optimal['rainfall'] - 30:
            analysis['issues'].append(f"⚠️ Rainfall insufficient ({rainfall}mm vs optimal {optimal['rainfall']}mm)")
        elif rainfall > optimal['rainfall'] + 30:
            analysis['issues'].append(f"⚠️ Rainfall excessive ({rainfall}mm vs optimal {optimal['rainfall']}mm)")
        
        return analysis
    
    def _calculate_feasibility(self, soil_analysis, env_analysis):
        """Calculate feasibility score (0-100)"""
        score = 100
        
        # Soil analysis penalties
        nutrient_issues = len([rec for rec in soil_analysis.get('recommendations', [])])
        score -= nutrient_issues * 10
        
        # Environmental issues
        env_issues = len(env_analysis.get('issues', []))
        score -= env_issues * 15
        
        return max(0, score)
    
    def _rate_feasibility(self, score):
        """Rate feasibility based on score"""
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 50:
            return 'Moderate'
        elif score >= 30:
            return 'Poor'
        else:
            return 'Not Recommended'
    
    def _generate_recommendation_text(self, crop_name, soil_analysis, env_analysis, fertilizer_rec):
        """Generate human-readable recommendation"""
        text = f"For {crop_name}:\n\n"
        
        # Soil recommendations
        if soil_analysis.get('recommendations'):
            text += "Soil Amendments Needed:\n"
            for rec in soil_analysis['recommendations']:
                text += f"  • {rec}\n"
            text += "\n"
        else:
            text += "✓ Soil conditions are optimal for this crop\n\n"
        
        # Environmental considerations
        if env_analysis.get('issues'):
            text += "Environmental Considerations:\n"
            for issue in env_analysis['issues']:
                text += f"  • {issue}\n"
            text += "\n"
        else:
            text += "✓ Environmental conditions are suitable\n\n"
        
        # Fertilizer recommendation
        text += f"Recommended Fertilizer: {fertilizer_rec['recommendation']}\n"
        text += f"Application Rate: {fertilizer_rec['application_rate']}\n"
        text += f"Reason: {fertilizer_rec['reason']}\n"
        
        return text


# Singleton instance
_recommendation_engine = None

def get_recommendation_engine():
    """Get or create recommendation engine"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine
