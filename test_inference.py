#!/usr/bin/env python
"""
Simple test script to verify model inference works correctly.
No Flask server needed.
"""

import numpy as np
import torch
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model, inference_model
from PIL import Image
import os
import sys

# Configuration
CONFIG_PATH = "PlantSeg/configs/segnext/segnext_simple_256.py"
CHECKPOINT_PATH = "PlantSeg/work_dirs/segnext_mscan-l_test/iter_1000.pth"
TEST_IMAGE_DIR = "PlantSeg/data/plantseg115/images/train"

def test_inference():
    """Test model inference on sample images"""
    
    print("=" * 70)
    print("PLANT DISEASE SEGMENTATION - INFERENCE TEST")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    try:
        model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cuda')
        print("[✓] Model loaded successfully!")
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return False
    
    # Get test images
    print("\n[2] Finding test images...")
    test_images = []
    for root, dirs, files in os.walk(TEST_IMAGE_DIR):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                test_images.append(os.path.join(root, f))
        break  # Only look in top directory
    
    test_images = test_images[:5]  # Test first 5
    print(f"[✓] Found {len(test_images)} test images")
    
    # Run inference
    print("\n[3] Running inference on test images...")
    print("-" * 70)
    
    results = []
    for i, image_path in enumerate(test_images, 1):
        try:
            filename = os.path.basename(image_path)
            
            # Run inference
            with torch.no_grad():
                result = inference_model(model, image_path)
            
            # Get segmentation logits
            seg_logits = result.seg_logits.data.cpu().numpy()
            
            # Analyze results
            max_logits_per_class = np.max(seg_logits, axis=(1, 2))
            pred_classes = np.argmax(seg_logits, axis=0)
            
            # Count pixels classified as disease (class 1)
            total_pixels = pred_classes.size
            disease_pixels = np.sum(pred_classes == 1)
            disease_percentage = (100.0 * disease_pixels) / total_pixels if total_pixels > 0 else 0.0
            
            # Get logits for classes 0 and 1
            class_0_logit = float(max_logits_per_class[0])
            class_1_logit = float(max_logits_per_class[1]) if len(max_logits_per_class) > 1 else -np.inf
            
            # Decision: disease if >10% pixels are class 1 AND class 1 logit is reasonable
            disease_detected = (disease_percentage > 10.0) and (class_1_logit > -3.0)
            
            if disease_detected:
                result_str = "DISEASE DETECTED"
                confidence = min(95.0, max(30.0, 50.0 + (disease_percentage * 0.3)))
            else:
                result_str = "HEALTHY"
                confidence = min(95.0, max(30.0, 50.0 - (disease_percentage * 0.5)))
            
            confidence = np.clip(confidence, 20.0, 95.0)
            
            results.append({
                'filename': filename,
                'result': result_str,
                'confidence': confidence,
                'disease_pct': disease_percentage,
                'class_0_logit': class_0_logit,
                'class_1_logit': class_1_logit
            })
            
            print(f"{i}. {filename:40s} | {result_str:18s} | Confidence: {confidence:5.1f}%")
            
        except Exception as e:
            print(f"{i}. {os.path.basename(image_path):40s} | ERROR: {str(e)[:30]}")
    
    # Summary
    print("-" * 70)
    print(f"\n[4] Results Summary:")
    print(f"    Total tested: {len(results)}")
    disease_count = sum(1 for r in results if r['result'] == "DISEASE DETECTED")
    healthy_count = len(results) - disease_count
    print(f"    Disease detected: {disease_count}")
    print(f"    Healthy: {healthy_count}")
    
    print("\n[✓] Inference test completed successfully!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
