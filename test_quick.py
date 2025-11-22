#!/usr/bin/env python
"""
Quick test script to validate intermediate checkpoints during training.
This can be run while training is still in progress.

Usage:
    python test_quick.py                  # Test latest checkpoint
    python test_quick.py --iter 20000     # Test specific iteration
    python test_quick.py --test-all       # Comprehensive test on all test images
"""

import os
import sys
from pathlib import Path
import argparse

# Add PlantSeg to path
sys.path.insert(0, 'PlantSeg')

import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from collections import Counter

def find_latest_checkpoint():
    """Find the latest checkpoint from training"""
    work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    
    if not work_dir.exists():
        return None
    
    checkpoints = sorted(work_dir.glob('iter_*.pth'))
    if checkpoints:
        return checkpoints[-1]
    
    return None

def test_checkpoint(checkpoint_path, num_images=5):
    """Test a single checkpoint on sample images"""
    
    print("\n" + "="*70)
    print("QUICK INFERENCE TEST")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Testing on {num_images} sample images")
    print("="*70)
    
    # Load model
    try:
        config_path = 'PlantSeg/configs/segnext/segnext_mscan-l_full_plantseg115-512x512.py'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n[*] Loading model on {device}...")
        model = init_model(config_path, str(checkpoint_path), device=device)
        model.eval()
        print("[OK] Model loaded!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    
    # Get test images
    test_dir = Path('PlantSeg/data/plantseg115/images/test')
    test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))[:num_images]
    
    if not test_images:
        print(f"[ERROR] No test images found in {test_dir}")
        return False
    
    print(f"\n[*] Running inference on {len(test_images)} images...")
    
    all_classes = set()
    results = []
    
    for idx, img_path in enumerate(test_images, 1):
        try:
            print(f"\n  [{idx}/{len(test_images)}] {img_path.name}")
            
            with torch.no_grad():
                result = inference_model(model, str(img_path))
            
            seg_logits = result.seg_logits.data.cpu().numpy()
            pred_classes = np.argmax(seg_logits, axis=0)
            
            unique_classes = set(np.unique(pred_classes).tolist())
            all_classes.update(unique_classes)
            
            # Count predictions
            class_counts = Counter(pred_classes.flatten())
            top_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            results.append({
                'image': img_path.name,
                'unique_classes': len(unique_classes),
                'top_class': top_class,
                'total_classes_so_far': len(all_classes)
            })
            
            print(f"      Unique classes: {len(unique_classes)}")
            print(f"      Top prediction: Class {top_class}")
            print(f"      Total unique across all: {len(all_classes)}")
            
        except Exception as e:
            print(f"      ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Images tested: {len(results)}")
    print(f"Unique disease classes predicted: {len(all_classes)}")
    
    if len(all_classes) >= 50:
        print(f"✓ Model predicts {len(all_classes)} disease classes (GOOD!)")
        return True
    elif len(all_classes) >= 10:
        print(f"△ Model predicts {len(all_classes)} disease classes (OK, but limited)")
        return True
    else:
        print(f"✗ Model predicts only {len(all_classes)} disease classes (needs more training)")
        return False

def main():
    parser = argparse.ArgumentParser(description='Quick inference test')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--iter', type=int, help='Specific iteration to test')
    parser.add_argument('--num-images', type=int, default=5, help='Number of test images')
    parser.add_argument('--test-all', action='store_true', help='Test on all test images')
    
    args = parser.parse_args()
    
    # Determine checkpoint to test
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.iter:
        work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
        checkpoint_path = work_dir / f'iter_{args.iter}.pth'
    else:
        checkpoint_path = find_latest_checkpoint()
    
    if not checkpoint_path or not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
        if work_dir.exists():
            for cp in sorted(work_dir.glob('iter_*.pth'))[-3:]:
                print(f"  - {cp}")
        sys.exit(1)
    
    # Run test
    num_images = 100 if args.test_all else args.num_images
    success = test_checkpoint(checkpoint_path, num_images)
    
    print("\n" + "="*70)
    if success:
        print("✓ Test completed successfully!")
    print("="*70)

if __name__ == '__main__':
    main()
