"""
Fertilizer Recommendation System
Uses rule-based classification to recommend fertilizers based on N, P, K deficiencies
Reference: Fertilizer Recommendation Dataset with custom rules
"""

import numpy as np
from enum import Enum

class FertilizerType(Enum):
    """Types of fertilizers available"""
    UREA = "Urea (46% N)"
    DAP = "DAP - Di-Ammonium Phosphate (18% N, 46% P)"
    POTASH = "Potassium Chloride (60% K)"
    NPK_10_10_10 = "NPK 10:10:10 (Balanced)"
    NPK_20_20_20 = "NPK 20:20:20 (High NPK)"
    NPK_15_15_15 = "NPK 15:15:15 (Medium)"
    AMMONIUM_NITRATE = "Ammonium Nitrate (35% N)"
    CALCIUM_NITRATE = "Calcium Nitrate (15.5% N, 26% Ca)"
    COMPOST = "Organic Compost"
    NO_FERTILIZER = "Soil already optimal - No fertilizer needed"


class FertilizerRecommender:
    """
    Rule-based fertilizer recommendation system
    Recommends fertilizer based on N, P, K deficiency levels
    """
    
    def __init__(self):
        """Initialize fertilizer recommender with deficiency thresholds"""
        self.low_threshold = 0.70  # 30% below optimal
        self.high_threshold = 1.30  # 30% above optimal
        
        # Deficiency levels: 0-30%, 30-60%, 60-90%, 90%+
        self.deficiency_severe = 0.30
        self.deficiency_moderate = 0.60
        self.deficiency_mild = 0.90
    
    def classify_deficiency(self, current, optimal):
        """
        Classify nutrient deficiency level
        
        Args:
            current: Current nutrient value
            optimal: Optimal nutrient value
        
        Returns:
            Tuple of (deficiency_level, deficit_percentage)
        """
        if current >= optimal * self.high_threshold:
            return ('excess', (current - optimal) / optimal * 100)
        elif current >= optimal * 0.90:
            return ('optimal', 0)
        elif current >= optimal * self.deficiency_mild:
            return ('mild_deficiency', (optimal - current) / optimal * 100)
        elif current >= optimal * self.deficiency_moderate:
            return ('moderate_deficiency', (optimal - current) / optimal * 100)
        else:
            return ('severe_deficiency', (optimal - current) / optimal * 100)
    
    def recommend(self, crop_name, current_N, current_P, current_K, optimal_N, optimal_P, optimal_K):
        """
        Recommend fertilizer based on NPK deficiency
        
        Args:
            crop_name: Name of crop
            current_N, P, K: Current soil nutrient values
            optimal_N, P, K: Optimal nutrient values for crop
        
        Returns:
            Dictionary with fertilizer recommendations
        """
        # Classify deficiencies
        n_status, n_deficit = self.classify_deficiency(current_N, optimal_N)
        p_status, p_deficit = self.classify_deficiency(current_P, optimal_P)
        k_status, k_deficit = self.classify_deficiency(current_K, optimal_K)
        
        # Check if soil is already optimal
        if n_status == 'optimal' and p_status == 'optimal' and k_status == 'optimal':
            return {
                'success': True,
                'crop': crop_name,
                'recommendation': FertilizerType.NO_FERTILIZER.value,
                'reason': 'Soil nutrient levels are optimal for the selected crop',
                'application_rate': 'None',
                'nutrient_status': {
                    'N': {'status': n_status, 'deficit': n_deficit, 'current': current_N, 'optimal': optimal_N},
                    'P': {'status': p_status, 'deficit': p_deficit, 'current': current_P, 'optimal': optimal_P},
                    'K': {'status': k_status, 'deficit': k_deficit, 'current': current_K, 'optimal': optimal_K}
                }
            }
        
        # Determine primary deficiency
        deficiencies = []
        if n_status != 'optimal':
            deficiencies.append(('N', n_status, n_deficit, current_N, optimal_N))
        if p_status != 'optimal':
            deficiencies.append(('P', p_status, p_deficit, current_P, optimal_P))
        if k_status != 'optimal':
            deficiencies.append(('K', k_status, k_deficit, current_K, optimal_K))
        
        # Rule-based recommendation logic
        recommendation = self._apply_rules(n_status, p_status, k_status, 
                                          n_deficit, p_deficit, k_deficit,
                                          current_N, current_P, current_K,
                                          optimal_N, optimal_P, optimal_K)
        
        return {
            'success': True,
            'crop': crop_name,
            'recommendation': recommendation['fertilizer'],
            'reason': recommendation['reason'],
            'application_rate': recommendation['application_rate'],
            'nutrient_status': {
                'N': {'status': n_status, 'deficit': n_deficit, 'current': current_N, 'optimal': optimal_N},
                'P': {'status': p_status, 'deficit': p_deficit, 'current': current_P, 'optimal': optimal_P},
                'K': {'status': k_status, 'deficit': k_deficit, 'current': current_K, 'optimal': optimal_K}
            },
            'deficiencies': deficiencies,
            'instructions': recommendation['instructions']
        }
    
    def _apply_rules(self, n_status, p_status, k_status, n_def, p_def, k_def, 
                     curr_n, curr_p, curr_k, opt_n, opt_p, opt_k):
        """Apply rule-based classification for fertilizer recommendation"""
        
        # Count severe deficiencies
        severe_count = sum([
            1 for status in [n_status, p_status, k_status] 
            if status == 'severe_deficiency'
        ])
        
        moderate_count = sum([
            1 for status in [n_status, p_status, k_status] 
            if status == 'moderate_deficiency'
        ])
        
        # Rule 1: All three nutrients severely deficient -> High NPK
        if severe_count >= 2:
            return {
                'fertilizer': FertilizerType.NPK_20_20_20.value,
                'reason': 'Multiple severe nutrient deficiencies detected',
                'application_rate': '50-75 kg/hectare',
                'instructions': [
                    'Apply 50-75 kg/hectare of NPK 20:20:20',
                    'Split application into 2-3 doses',
                    'Apply 1st dose at planting, 2nd at flowering',
                    'Water adequately after each application'
                ]
            }
        
        # Rule 2: Moderate deficiencies in multiple nutrients -> Balanced NPK
        if moderate_count >= 2 or (n_status in ['moderate_deficiency', 'severe_deficiency'] and 
                                   p_status in ['moderate_deficiency', 'severe_deficiency'] and 
                                   k_status in ['moderate_deficiency', 'severe_deficiency']):
            return {
                'fertilizer': FertilizerType.NPK_15_15_15.value,
                'reason': 'Moderate deficiencies in multiple nutrients',
                'application_rate': '40-60 kg/hectare',
                'instructions': [
                    'Apply 40-60 kg/hectare of NPK 15:15:15',
                    'Split into 2 doses: at planting and mid-season',
                    'Ensure proper soil moisture before application',
                    'Incorporate fertilizer into soil'
                ]
            }
        
        # Rule 3: Primarily Nitrogen deficient
        if n_status in ['severe_deficiency', 'moderate_deficiency'] and \
           p_status in ['optimal', 'excess'] and k_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.UREA.value,
                'reason': f'Primary Nitrogen deficiency ({n_def:.1f}% below optimal)',
                'application_rate': f'{max(25, int(opt_n - curr_n))}-40 kg/hectare',
                'instructions': [
                    f'Apply {max(25, int(opt_n - curr_n))}-40 kg/hectare of Urea',
                    'Split into 2-3 doses during growing season',
                    'Apply 1st dose at 4-5 leaf stage, then at tillering',
                    'Water thoroughly after application'
                ]
            }
        
        # Rule 4: Primarily Phosphorus deficient
        if p_status in ['severe_deficiency', 'moderate_deficiency'] and \
           n_status in ['optimal', 'excess'] and k_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.DAP.value,
                'reason': f'Primary Phosphorus deficiency ({p_def:.1f}% below optimal)',
                'application_rate': f'{max(20, int(opt_p - curr_p))}-35 kg/hectare',
                'instructions': [
                    f'Apply {max(20, int(opt_p - curr_p))}-35 kg/hectare of DAP',
                    'Best applied at planting time',
                    'Mix with soil during field preparation',
                    'DAP also provides Nitrogen (18%)'
                ]
            }
        
        # Rule 5: Primarily Potassium deficient
        if k_status in ['severe_deficiency', 'moderate_deficiency'] and \
           n_status in ['optimal', 'excess'] and p_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.POTASH.value,
                'reason': f'Primary Potassium deficiency ({k_def:.1f}% below optimal)',
                'application_rate': f'{max(20, int(opt_k - curr_k))}-40 kg/hectare',
                'instructions': [
                    f'Apply {max(20, int(opt_k - curr_k))}-40 kg/hectare of Potassium Chloride',
                    'Best applied during mid-growing season',
                    'Apply in 2 split doses',
                    'Essential for fruit/grain development'
                ]
            }
        
        # Rule 6: N and P deficient, K optimal
        if n_status in ['severe_deficiency', 'moderate_deficiency'] and \
           p_status in ['severe_deficiency', 'moderate_deficiency'] and \
           k_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.DAP.value,
                'reason': 'Nitrogen and Phosphorus deficiency',
                'application_rate': '40-60 kg/hectare',
                'instructions': [
                    'Apply 40-60 kg/hectare of DAP (18% N, 46% P)',
                    'Apply at planting for best results',
                    'Also consider split Urea application for additional N',
                    'Follow with Urea 2-3 weeks after planting'
                ]
            }
        
        # Rule 7: N and K deficient, P optimal
        if n_status in ['severe_deficiency', 'moderate_deficiency'] and \
           k_status in ['severe_deficiency', 'moderate_deficiency'] and \
           p_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.NPK_10_10_10.value,
                'reason': 'Nitrogen and Potassium deficiency',
                'application_rate': '50-75 kg/hectare',
                'instructions': [
                    'Apply 50-75 kg/hectare of NPK 10:10:10',
                    'Split into 2 doses during growing season',
                    'Apply 1st dose at planting, 2nd at mid-season',
                    'Ensure adequate irrigation'
                ]
            }
        
        # Rule 8: P and K deficient, N optimal
        if p_status in ['severe_deficiency', 'moderate_deficiency'] and \
           k_status in ['severe_deficiency', 'moderate_deficiency'] and \
           n_status in ['optimal', 'excess']:
            return {
                'fertilizer': FertilizerType.NPK_10_10_10.value,
                'reason': 'Phosphorus and Potassium deficiency',
                'application_rate': '50-75 kg/hectare',
                'instructions': [
                    'Apply 50-75 kg/hectare of NPK 10:10:10',
                    'Best applied at planting time',
                    'Mix into soil before planting',
                    'Provide split Potassium dose at mid-season'
                ]
            }
        
        # Rule 9: Mild deficiency in one nutrient
        if (n_status == 'mild_deficiency' or p_status == 'mild_deficiency' or k_status == 'mild_deficiency') and \
           sum([1 for s in [n_status, p_status, k_status] if s in ['moderate_deficiency', 'severe_deficiency']]) == 0:
            return {
                'fertilizer': FertilizerType.COMPOST.value,
                'reason': 'Mild nutrient deficiency - organic amendment recommended',
                'application_rate': '5-10 tons/hectare',
                'instructions': [
                    'Apply 5-10 tons/hectare of well-decomposed compost',
                    'Incorporate into top 10-15 cm of soil',
                    'Apply 2-4 weeks before planting',
                    'Improves soil structure and nutrient availability'
                ]
            }
        
        # Default: Balanced NPK
        return {
            'fertilizer': FertilizerType.NPK_10_10_10.value,
            'reason': 'Mixed nutrient profile - balanced NPK recommended',
            'application_rate': '40-50 kg/hectare',
            'instructions': [
                'Apply 40-50 kg/hectare of NPK 10:10:10',
                'Apply at planting or early growing stage',
                'Follow with targeted applications based on growth stage',
                'Monitor soil and plant health regularly'
            ]
        }
    
    def get_fertilizer_info(self, fertilizer_type):
        """Get detailed information about a fertilizer type"""
        info = {
            FertilizerType.UREA.value: {
                'npk': '46-0-0',
                'cost': 'Low',
                'best_for': 'Nitrogen deficiency',
                'timing': 'Mid-season',
                'water_needed': 'High',
                'pros': 'Fast acting, high N content, affordable',
                'cons': 'Easily leachable, requires split application'
            },
            FertilizerType.DAP.value: {
                'npk': '18-46-0',
                'cost': 'Medium',
                'best_for': 'Nitrogen and Phosphorus deficiency',
                'timing': 'At planting',
                'water_needed': 'Medium',
                'pros': 'Dual nutrient, easy to apply, balanced',
                'cons': 'More expensive than single nutrient'
            },
            FertilizerType.POTASH.value: {
                'npk': '0-0-60',
                'cost': 'Medium',
                'best_for': 'Potassium deficiency',
                'timing': 'Mid to late season',
                'water_needed': 'Medium',
                'pros': 'High K content, fruit quality',
                'cons': 'Requires split application'
            },
            FertilizerType.NPK_10_10_10.value: {
                'npk': '10-10-10',
                'cost': 'Medium',
                'best_for': 'Balanced nutrition',
                'timing': 'At planting and mid-season',
                'water_needed': 'Medium',
                'pros': 'Balanced, versatile, suitable for most crops',
                'cons': 'Lower NPK content than specialized'
            },
            FertilizerType.NPK_20_20_20.value: {
                'npk': '20-20-20',
                'cost': 'High',
                'best_for': 'Severe multiple deficiencies',
                'timing': 'Early and mid-season',
                'water_needed': 'High',
                'pros': 'High nutrient content, fast acting',
                'cons': 'Expensive, risk of overdose'
            }
        }
        return info.get(fertilizer_type, {})


# Singleton instance
_fertilizer_recommender = None

def get_fertilizer_recommender():
    """Get or create fertilizer recommender"""
    global _fertilizer_recommender
    if _fertilizer_recommender is None:
        _fertilizer_recommender = FertilizerRecommender()
    return _fertilizer_recommender
