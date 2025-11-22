"""
Recommendation AI Package
Crop and Fertilizer Recommendation System

Components:
- crop_recommender.py: Recommends crops based on soil/environmental conditions
- fertilizer_recommender.py: Recommends fertilizers based on NPK deficiency
- recommendation_engine.py: Integrated engine combining both recommendations
"""


from .crop_recommender import CropRecommender, get_crop_recommender
from .fertilizer_recommender import FertilizerRecommender, get_fertilizer_recommender
from .recommendation_engine import RecommendationEngine, get_recommendation_engine

__all__ = [
    'CropRecommender',
    'FertilizerRecommender',
    'RecommendationEngine',
    'get_crop_recommender',
    'get_fertilizer_recommender',
    'get_recommendation_engine'
]

__version__ = '1.0.0'
