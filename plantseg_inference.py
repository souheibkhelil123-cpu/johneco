"""
PlantSeg Inference Module
Handles loading and running the PlantSeg segmentation model
"""

import os
import sys
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import io

warnings.filterwarnings("ignore")

# Add PlantSeg to path
PLANTSEG_PATH = Path(__file__).parent / "PlantSeg"
sys.path.insert(0, str(PLANTSEG_PATH))

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model, inference_model
from mmseg.registry import MODELS
from mmengine.registry import init_default_scope


class PlantSegInferencer:
    """
    PlantSeg segmentation model wrapper for plant organ segmentation
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PlantSeg model
        
        Args:
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            device: Device to run model on (cuda:0, cuda:1, cpu, etc.)
        """
        self.device = device
        self.model = None
        self.config = None
        self.checkpoint_path = checkpoint_path
        
        # Default to deeplabv3 plantseg model
        if config_path is None:
            config_path = str(PLANTSEG_PATH / "configs" / "deeplabv3" / "deeplabv3_r101-160k_plantseg_binary-128x128.py")
        
        self.config_path = config_path
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the segmentation model"""
        try:
            print(f"Loading PlantSeg model from config: {self.config_path}")
            print(f"Using device: {self.device}")
            
            # Initialize model with config
            self.model = init_model(
                self.config_path,
                checkpoint=self.checkpoint_path,
                device=self.device
            )
            
            print("✓ PlantSeg model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            self.model = None
            return False
    
    def _preprocess_image(self, image_data: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_data: Input image as numpy array (RGB)
            target_size: Target size for model input
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if image_data.shape[:2] != target_size:
            image_data = cv2.resize(image_data, (target_size[1], target_size[0]))
        
        # Ensure RGB format
        if len(image_data.shape) == 2:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        elif image_data.shape[2] == 4:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
        
        return image_data.astype(np.uint8)
    
    def segment_image(self, image_data: np.ndarray) -> Dict:
        """
        Segment a single image
        
        Args:
            image_data: Input image as numpy array (RGB)
            
        Returns:
            Dictionary with segmentation results
        """
        if self.model is None:
            return {'error': 'Model not initialized'}
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            
            # Run inference
            result = inference_model(self.model, processed_image)
            
            # Extract segmentation mask
            seg_map = result.pred_sem_seg.data.cpu().numpy()
            
            # Get class probabilities if available
            confidence_map = None
            if hasattr(result, 'seg_logits'):
                confidence_map = torch.nn.functional.softmax(result.seg_logits, dim=0).cpu().numpy()
            
            # Create result dictionary
            result_dict = {
                'success': True,
                'segmentation_mask': seg_map,
                'confidence_map': confidence_map,
                'num_classes': self.model.decode_head.num_classes,
                'class_names': getattr(self.model, 'CLASSES', None),
                'palette': getattr(self.model, 'PALETTE', None),
            }
            
            return result_dict
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def segment_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Segment multiple images
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of segmentation results
        """
        results = []
        for idx, img in enumerate(images):
            try:
                result = self.segment_image(img)
                results.append(result)
            except Exception as e:
                results.append({'success': False, 'error': str(e), 'index': idx})
        
        return results
    
    def visualize_segmentation(self, image: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Create visualization of segmentation result
        
        Args:
            image: Original image
            seg_mask: Segmentation mask
            alpha: Transparency of overlay
            
        Returns:
            Visualization image
        """
        # Create colored mask
        color_mask = np.zeros_like(image)
        
        # Color different plant organs differently
        unique_classes = np.unique(seg_mask)
        colors = {
            0: [0, 0, 0],        # Background (black)
            1: [0, 255, 0],      # Plant/leaves (green)
            2: [139, 69, 19],    # Stem (brown)
            3: [255, 0, 0],      # Roots (red)
        }
        
        for class_id in unique_classes:
            color = colors.get(class_id, [np.random.randint(0, 256) for _ in range(3)])
            color_mask[seg_mask == class_id] = color
        
        # Blend with original image
        vis_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        
        return vis_image
    
    @staticmethod
    def load_image_from_file(file_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            img = cv2.imread(file_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def load_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
        """Load image from bytes (file upload)"""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            return np.array(image.convert('RGB'))
        except Exception as e:
            print(f"Error loading image from bytes: {str(e)}")
            return None
    
    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        """Convert image to base64 string for web display"""
        try:
            # Convert RGB to BGR for cv2
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', image_bgr)
            import base64
            return base64.b64encode(buffer).decode()
        except Exception as e:
            print(f"Error converting to base64: {str(e)}")
            return ""


# Lazy loading for efficiency
_inferencer = None

def get_inferencer(force_reload: bool = False) -> PlantSegInferencer:
    """Get or create PlantSeg inferencer (lazy loading)"""
    global _inferencer
    
    if _inferencer is None or force_reload:
        _inferencer = PlantSegInferencer()
    
    return _inferencer


if __name__ == "__main__":
    # Test the module
    print("Testing PlantSeg Inference Module...")
    inferencer = get_inferencer()
    
    # Try to segment a test image
    test_image_path = PLANTSEG_PATH / "image" / "test.txt"
    if test_image_path.exists():
        print(f"Found test data reference: {test_image_path}")
    else:
        print("No test images found, but module is ready for use")
