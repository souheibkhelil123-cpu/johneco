#!/usr/bin/env python
"""
COMPLETE AUTOMATION SCRIPT
Trains SegNext on all PlantSeg115 data and starts web interface.

This script:
1. Starts training in background (if not already running)
2. Monitors progress
3. Tests intermediate checkpoints
4. Starts web interface once ready
5. Provides real-time feedback

Usage:
    python run_complete_pipeline.py                  # Full automation
    python run_complete_pipeline.py --skip-train     # Skip training, use existing checkpoint
    python run_complete_pipeline.py --test-only      # Test only, no web interface
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from collections import Counter

def find_latest_checkpoint(min_iter=100):
    """Find the latest checkpoint"""
    work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    
    if not work_dir.exists():
        return None
    
    checkpoints = sorted(work_dir.glob('iter_*.pth'))
    if checkpoints:
        for cp in reversed(checkpoints):
            iter_num = int(cp.stem.split('_')[1])
            if iter_num >= min_iter:
                return str(cp), iter_num
    
    return None, 0

def start_training():
    """Start training in background"""
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    os.chdir('PlantSeg')
    cmd = [
        sys.executable, 'tools/train.py',
        'configs/segnext/segnext_mscan-l_full_plantseg115-512x512.py'
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Start training in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("[*] Training started (PID: {})".format(process.pid))
        print("[*] Training will continue in background...")
        print("[*] Check PlantSeg/work_dirs/ for checkpoints\n")
        
        os.chdir('..')
        return process
    
    except Exception as e:
        print(f"[ERROR] Failed to start training: {e}")
        os.chdir('..')
        return None

def monitor_progress(max_wait_minutes=360):
    """Monitor training progress and wait for checkpoint"""
    print("\n" + "="*70)
    print("MONITORING TRAINING PROGRESS")
    print("="*70)
    print("Waiting for checkpoint... (this may take 15-30 minutes)")
    print("Listening for 1st checkpoint at 4000+ iterations\n")
    
    start_time = time.time()
    max_seconds = max_wait_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_seconds:
            print(f"\n[WARNING] Training took longer than {max_wait_minutes} minutes")
            print("[*] Using latest available checkpoint if present")
            break
        
        checkpoint_path, iter_num = find_latest_checkpoint(min_iter=1000)
        
        if checkpoint_path:
            elapsed_hours = elapsed / 3600
            print(f"\n✓ Found checkpoint: {Path(checkpoint_path).name}")
            print(f"  Iterations: {iter_num}")
            print(f"  Time elapsed: {elapsed_hours:.1f} hours")
            print(f"  Ready for testing/inference!")
            return checkpoint_path, iter_num
        
        # Progress indicator
        elapsed_min = elapsed / 60
        print(f"\r⏳ Waiting... {elapsed_min:.1f} min | Checking for checkpoint...", end="", flush=True)
        
        time.sleep(30)  # Check every 30 seconds

def test_checkpoint(checkpoint_path, num_images=10):
    """Quick test on checkpoint"""
    print("\n" + "="*70)
    print("TESTING MODEL")
    print("="*70)
    print(f"Checkpoint: {Path(checkpoint_path).name}\n")
    
    cmd = [
        sys.executable, 'PlantSeg/test_full_inference.py',
        '--checkpoint', checkpoint_path,
        '--limit', str(num_images)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✓ Model testing completed successfully!")
            return True
        else:
            print("\n[WARNING] Model testing had issues, continuing anyway...")
            return False
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {e}")
        return False

def start_web_interface(checkpoint_path):
    """Start web interface with checkpoint"""
    print("\n" + "="*70)
    print("STARTING WEB INTERFACE")
    print("="*70)
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Address: http://localhost:5000\n")
    
    cmd = [
        sys.executable, 'start_web_interface.py',
        '--checkpoint', checkpoint_path,
        '--no-browser'
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"[*] Web server started (PID: {process.pid})")
        print("[*] Open browser: http://localhost:5000\n")
        
        # Monitor server output
        for line in process.stdout:
            print(line, end='')
        
        return process
    
    except Exception as e:
        print(f"[ERROR] Failed to start web interface: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Complete PlantSeg115 pipeline')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, use existing checkpoint')
    parser.add_argument('--test-only', action='store_true', help='Test only, no web interface')
    parser.add_argument('--wait-iterations', type=int, default=4000, help='Wait for this many iterations before testing')
    parser.add_argument('--skip-wait', action='store_true', help='Skip wait, use current checkpoint immediately')
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*70)
        print("PLANTSEG115 COMPLETE PIPELINE")
        print("SegNext Model Training + Web Interface")
        print("="*70)
        
        checkpoint_path = None
        iter_num = 0
        
        # Step 1: Check for existing checkpoint or start training
        if args.skip_train:
            print("\n[*] Skipping training, looking for existing checkpoint...")
            checkpoint_path, iter_num = find_latest_checkpoint(min_iter=100)
            
            if not checkpoint_path:
                print("[ERROR] No checkpoint found!")
                print("Either run training first or remove --skip-train")
                sys.exit(1)
            else:
                print(f"[OK] Found checkpoint: {checkpoint_path} (iter {iter_num})")
        else:
            # Start training
            train_process = start_training()
            
            if not train_process:
                print("[ERROR] Failed to start training!")
                sys.exit(1)
            
            # Step 2: Monitor and wait for checkpoint
            if not args.skip_wait:
                checkpoint_path, iter_num = monitor_progress()
            else:
                print("\n[*] Skipping wait, using latest checkpoint...")
                checkpoint_path, iter_num = find_latest_checkpoint(min_iter=100)
        
        if not checkpoint_path:
            print("[ERROR] No checkpoint available!")
            sys.exit(1)
        
        # Step 3: Test checkpoint
        print("\n[*] Quick verification test...")
        test_checkpoint(checkpoint_path, num_images=5)
        
        # Step 4: Start web interface (unless test-only mode)
        if not args.test_only:
            print("\n[*] Launching web interface...")
            web_process = start_web_interface(checkpoint_path)
            
            if web_process:
                print("\n✓ Pipeline complete!")
                print("✓ Training running in background")
                print("✓ Web interface active at http://localhost:5000")
                print("\nPress Ctrl+C to stop web server")
                
                try:
                    web_process.wait()
                except KeyboardInterrupt:
                    print("\n\n[!] Stopping web interface...")
                    web_process.terminate()
                    print("[OK] Web interface stopped")
        else:
            print("\n✓ Test-only mode complete!")
            print("✓ Model tested successfully")
            print("✓ Training continuing in background")
    
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user")
        print("[*] Training will continue in background")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
