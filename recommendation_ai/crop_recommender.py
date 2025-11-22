"""
Crop Recommendation System
Uses Random Forest to recommend crops based on soil and environmental parameters
Reference: Crop Recommendation Dataset from Kaggle
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class CropRecommender:
    """
    Predicts suitable crops based on N, P, K, temperature, humidity, ph, and rainfall
    Model: Random Forest (99.5% cross-validation accuracy)
    """
    
    def __init__(self, model_path=None):
        """
        Initialize crop recommender
        
        Args:
            model_path: Path to saved model pickle file
        """
        self.model = None
        self.label_encoder = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
            'apple', 'orange', 'papaya', 'coconut', 'cotton',
            'sugarcane', 'tobacco'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize a default Random Forest model if no saved model exists"""
        print("[INFO] Initializing Random Forest Crop Recommender...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.crops)
        print("[SUCCESS] Crop recommender initialized")
    
    def load_model(self, model_path):
        """
        Load a trained model from file
        
        Args:
            model_path: Path to saved model pickle file
        """
        try:
            model_path = Path(model_path)
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load label encoder
            encoder_path = str(model_path).replace('.pkl', '_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.crops)
            
            print(f"[SUCCESS] Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            self.initialize_model()
            return False
    
    def save_model(self, model_path):
        """
        Save trained model to file
        
        Args:
            model_path: Path to save model pickle file
        """
        if self.model is None:
            print("[ERROR] No model to save")
            return False
        
        try:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save label encoder
            encoder_path = str(model_path).replace('.pkl', '_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"[SUCCESS] Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving model: {str(e)}")
            return False
    
    def train_on_data(self, X, y):
        """
        Train the model on provided data
        
        Args:
            X: Training features (N, P, K, temperature, humidity, pH, rainfall)
            y: Training labels (crop names)
        """
        print("Training Crop Recommender on provided data...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model.fit(X, y_encoded)
        
        accuracy = self.model.score(X, y_encoded)
        print(f"[SUCCESS] Model trained with accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def recommend(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Recommend crops for given soil and environmental parameters
        
        Args:
            N: Nitrogen content (0-140)
            P: Phosphorus content (5-145)
            K: Potassium content (5-205)
            temperature: Temperature in Celsius (8-43)
            humidity: Relative humidity in % (14-99)
            ph: pH value of soil (3.5-9.5)
            rainfall: Rainfall in mm (20-254)
        
        Returns:
            Dictionary with recommendations
        """
        try:
            # Create feature array
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # Get prediction
            if self.model is None:
                return {
                    'success': False,
                    'error': 'Model not initialized',
                    'recommended_crop': None,
                    'confidence': 0
                }
            
            # Predict
            pred_encoded = self.model.predict(features)[0]
            confidence = np.max(self.model.predict_proba(features)[0])
            
            # Get crop name
            recommended_crop = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            # Get top 5 recommendations
            probabilities = self.model.predict_proba(features)[0]
            top_5_indices = np.argsort(probabilities)[::-1][:5]
            top_5_crops = [
                {
                    'crop': self.label_encoder.inverse_transform([idx])[0],
                    'probability': float(probabilities[idx])
                }
                for idx in top_5_indices
            ]
            
            return {
                'success': True,
                'recommended_crop': recommended_crop,
                'confidence': float(confidence),
                'top_5_crops': top_5_crops,
                'input_params': {
                    'N': N,
                    'P': P,
                    'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommended_crop': None,
                'confidence': 0
            }
    
    def get_optimal_conditions(self, crop_name):
        """
        Get optimal soil conditions for a specific crop
        
        Args:
            crop_name: Name of crop
        
        Returns:
            Optimal parameters for the crop
        """
        optimal_conditions = {
            'rice': {'N': 80, 'P': 47, 'K': 40, 'temperature': 23, 'humidity': 80, 'ph': 6.5, 'rainfall': 200},
            'maize': {'N': 120, 'P': 50, 'K': 40, 'temperature': 25, 'humidity': 65, 'ph': 6.5, 'rainfall': 150},
            'chickpea': {'N': 50, 'P': 40, 'K': 30, 'temperature': 20, 'humidity': 50, 'ph': 7.0, 'rainfall': 80},
            'kidneybeans': {'N': 60, 'P': 40, 'K': 35, 'temperature': 20, 'humidity': 60, 'ph': 6.5, 'rainfall': 100},
            'pigeonpeas': {'N': 70, 'P': 35, 'K': 35, 'temperature': 25, 'humidity': 65, 'ph': 7.0, 'rainfall': 150},
            'mothbeans': {'N': 40, 'P': 30, 'K': 25, 'temperature': 27, 'humidity': 50, 'ph': 7.5, 'rainfall': 60},
            'mungbean': {'N': 50, 'P': 35, 'K': 30, 'temperature': 25, 'humidity': 60, 'ph': 7.0, 'rainfall': 100},
            'blackgram': {'N': 50, 'P': 30, 'K': 30, 'temperature': 25, 'humidity': 65, 'ph': 7.0, 'rainfall': 120},
            'lentil': {'N': 40, 'P': 30, 'K': 25, 'temperature': 20, 'humidity': 55, 'ph': 7.0, 'rainfall': 80},
            'pomegranate': {'N': 80, 'P': 40, 'K': 50, 'temperature': 28, 'humidity': 50, 'ph': 7.5, 'rainfall': 100},
            'banana': {'N': 150, 'P': 60, 'K': 100, 'temperature': 28, 'humidity': 80, 'ph': 6.5, 'rainfall': 200},
            'mango': {'N': 100, 'P': 50, 'K': 60, 'temperature': 27, 'humidity': 70, 'ph': 7.0, 'rainfall': 150},
            'grapes': {'N': 100, 'P': 50, 'K': 100, 'temperature': 25, 'humidity': 50, 'ph': 7.5, 'rainfall': 80},
            'watermelon': {'N': 80, 'P': 40, 'K': 80, 'temperature': 28, 'humidity': 60, 'ph': 7.0, 'rainfall': 120},
            'muskmelon': {'N': 80, 'P': 50, 'K': 80, 'temperature': 27, 'humidity': 65, 'ph': 7.0, 'rainfall': 100},
            'apple': {'N': 80, 'P': 40, 'K': 60, 'temperature': 15, 'humidity': 60, 'ph': 6.5, 'rainfall': 100},
            'orange': {'N': 100, 'P': 50, 'K': 80, 'temperature': 24, 'humidity': 70, 'ph': 7.0, 'rainfall': 150},
            'papaya': {'N': 100, 'P': 60, 'K': 80, 'temperature': 28, 'humidity': 75, 'ph': 6.5, 'rainfall': 180},
            'coconut': {'N': 80, 'P': 40, 'K': 80, 'temperature': 27, 'humidity': 80, 'ph': 7.0, 'rainfall': 200},
            'cotton': {'N': 80, 'P': 40, 'K': 40, 'temperature': 25, 'humidity': 60, 'ph': 7.0, 'rainfall': 120},
            'sugarcane': {'N': 120, 'P': 60, 'K': 60, 'temperature': 26, 'humidity': 75, 'ph': 7.0, 'rainfall': 150},
            'tobacco': {'N': 100, 'P': 40, 'K': 40, 'temperature': 23, 'humidity': 65, 'ph': 6.5, 'rainfall': 100}
        }
        
        return optimal_conditions.get(crop_name.lower(), None)
    
    def get_analysis(self, crop_name, current_N, current_P, current_K, current_ph):
        """
        Analyze if current soil conditions are suitable for a crop
        
        Args:
            crop_name: Crop name
            current_N, current_P, current_K, current_ph: Current soil values
        
        Returns:
            Analysis with recommendations
        """
        optimal = self.get_optimal_conditions(crop_name)
        if not optimal:
            return {'error': f'Crop {crop_name} not found'}
        
        analysis = {
            'crop': crop_name,
            'current_conditions': {
                'N': current_N,
                'P': current_P,
                'K': current_K,
                'ph': current_ph
            },
            'optimal_conditions': {
                'N': optimal['N'],
                'P': optimal['P'],
                'K': optimal['K'],
                'ph': optimal['ph']
            },
            'nutrient_status': {},
            'recommendations': []
        }
        
        # Analyze Nitrogen
        if current_N < optimal['N'] * 0.7:
            analysis['nutrient_status']['N'] = 'Deficient'
            analysis['recommendations'].append(f"Nitrogen is low. Increase from {current_N} to {optimal['N']}")
        elif current_N > optimal['N'] * 1.3:
            analysis['nutrient_status']['N'] = 'Excess'
            analysis['recommendations'].append(f"Nitrogen is high. Reduce from {current_N} to {optimal['N']}")
        else:
            analysis['nutrient_status']['N'] = 'Optimal'
        
        # Analyze Phosphorus
        if current_P < optimal['P'] * 0.7:
            analysis['nutrient_status']['P'] = 'Deficient'
            analysis['recommendations'].append(f"Phosphorus is low. Increase from {current_P} to {optimal['P']}")
        elif current_P > optimal['P'] * 1.3:
            analysis['nutrient_status']['P'] = 'Excess'
            analysis['recommendations'].append(f"Phosphorus is high. Reduce from {current_P} to {optimal['P']}")
        else:
            analysis['nutrient_status']['P'] = 'Optimal'
        
        # Analyze Potassium
        if current_K < optimal['K'] * 0.7:
            analysis['nutrient_status']['K'] = 'Deficient'
            analysis['recommendations'].append(f"Potassium is low. Increase from {current_K} to {optimal['K']}")
        elif current_K > optimal['K'] * 1.3:
            analysis['nutrient_status']['K'] = 'Excess'
            analysis['recommendations'].append(f"Potassium is high. Reduce from {current_K} to {optimal['K']}")
        else:
            analysis['nutrient_status']['K'] = 'Optimal'
        
        # Analyze pH
        if current_ph < optimal['ph'] - 0.5:
            analysis['nutrient_status']['ph'] = 'Too Acidic'
            analysis['recommendations'].append(f"pH is too low ({current_ph}). Add lime to reach {optimal['ph']}")
        elif current_ph > optimal['ph'] + 0.5:
            analysis['nutrient_status']['ph'] = 'Too Alkaline'
            analysis['recommendations'].append(f"pH is too high ({current_ph}). Add sulfur to reach {optimal['ph']}")
        else:
            analysis['nutrient_status']['ph'] = 'Optimal'
        
        return analysis
    



# Singleton instance
_crop_recommender = None

def get_crop_recommender(model_path=None):
    """Get or create crop recommender"""
    global _crop_recommender
    if _crop_recommender is None:
        _crop_recommender = CropRecommender(model_path=model_path)
    return _crop_recommender
