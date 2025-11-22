#!/usr/bin/env python
"""
Start the web interface for SegNext PlantSeg115 disease detection.

This script:
1. Waits for model training to complete OR uses latest checkpoint
2. Loads the trained model
3. Starts Flask web server on http://localhost:5000

Usage:
    python start_web_interface.py                    # Use latest checkpoint
    python start_web_interface.py --checkpoint <path>  # Use specific checkpoint
    python start_web_interface.py --wait             # Wait for training to complete
"""

import os
import sys
import time
import argparse
from pathlib import Path

def wait_for_training():
    """Wait until training reaches a good checkpoint (40k+ iterations)"""
    work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    target_iter = 40000
    
    print("\n" + "="*70)
    print("WAITING FOR TRAINING...")
    print("="*70)
    print(f"Target: {target_iter}+ iterations")
    print("(You can use Ctrl+C to skip wait and use current checkpoint)")
    
    try:
        while True:
            checkpoints = sorted(work_dir.glob('iter_*.pth'))
            
            if checkpoints:
                latest = checkpoints[-1]
                iter_num = int(latest.stem.split('_')[1])
                progress = (iter_num / 80000) * 100
                
                print(f"\rProgress: {iter_num}/{80000} ({progress:.1f}%)", end="", flush=True)
                
                if iter_num >= target_iter:
                    print(f"\n\n[OK] Training reached {iter_num} iterations!")
                    print(f"[OK] Using checkpoint: {latest}")
                    return str(latest)
            
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        print("\n\n[*] Skipping wait, using current checkpoint...")
        if checkpoints:
            return str(checkpoints[-1])
        else:
            print("[ERROR] No checkpoint found!")
            return None

def find_best_checkpoint(min_iter=1000):
    """Find best available checkpoint"""
    work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    
    if not work_dir.exists():
        return None
    
    checkpoints = sorted(work_dir.glob('iter_*.pth'))
    
    if checkpoints:
        # Get latest checkpoint with at least min_iter
        for cp in reversed(checkpoints):
            iter_num = int(cp.stem.split('_')[1])
            if iter_num >= min_iter:
                return str(cp)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Start web interface for disease detection')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to use')
    parser.add_argument('--wait', action='store_true', help='Wait for training to reach 40k iterations')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    
    args = parser.parse_args()
    
    # Determine checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.wait:
        checkpoint_path = wait_for_training()
        if not checkpoint_path:
            print("[ERROR] No checkpoint available!")
            sys.exit(1)
    else:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            print("[ERROR] No checkpoint found!")
            print("Training directory: PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115/")
            sys.exit(1)
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("STARTING WEB INTERFACE")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: SegNext MSCAN-L (115 disease classes)")
    print("="*70)
    
    # Import and run Flask app
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Set checkpoint as env var for web interface to use
    os.environ['SEGNEXT_CHECKPOINT'] = str(checkpoint_path)
    
    print("\n[*] Loading model...")
    print("[*] Starting Flask server on http://localhost:5000")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        from web_interface_full import app, load_model
        
        if load_model():
            print("\n✓ Model loaded successfully!")
            print("✓ Server ready!\n")
            
            if not args.no_browser:
                try:
                    import webbrowser
                    print("[*] Opening browser...")
                    webbrowser.open('http://localhost:5000')
                except:
                    print("[*] Please open http://localhost:5000 in your browser")
            
            # Run server
            app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        else:
            print("[ERROR] Failed to load model!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n[!] Server stopped")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
