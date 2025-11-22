#!/usr/bin/env python
"""
Monitor training progress and generate status report.
"""

import os
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor SegNext training progress"""
    
    work_dir = Path('PlantSeg/work_dirs/segnext_mscan-l_full_plantseg115')
    
    print("\n" + "="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)
    
    if not work_dir.exists():
        print("[ERROR] Training directory not found!")
        return
    
    # Find all checkpoints
    checkpoints = sorted(work_dir.glob('iter_*.pth'))
    
    if checkpoints:
        print(f"\n[OK] Training in progress")
        print(f"Latest checkpoint: {checkpoints[-1].name}")
        
        # Extract iteration number
        latest_iter = int(checkpoints[-1].stem.split('_')[1])
        total_iters = 80000
        progress = (latest_iter / total_iters) * 100
        
        print(f"\nProgress: {latest_iter}/{total_iters} iterations ({progress:.1f}%)")
        print(f"Checkpoints saved: {len(checkpoints)}")
        
        # Show checkpoint timeline
        print(f"\nCheckpoint timeline:")
        for i, cp in enumerate(checkpoints[-5:], 1):
            iter_num = int(cp.stem.split('_')[1])
            size_mb = cp.stat().st_size / (1024*1024)
            print(f"  {i}. {cp.name} ({size_mb:.1f} MB)")
        
        # Estimate time remaining
        if latest_iter > 1000:
            # Calculate approximate time per 1000 iterations
            # Assuming ~10-15 min per checkpoint (4000 iter chunks)
            iters_remaining = total_iters - latest_iter
            estimated_time = (iters_remaining / 1000) * 1.5  # ~90 sec per 1000 iter
            hours_remaining = estimated_time / 60
            print(f"\nEstimated time remaining: ~{hours_remaining:.1f} hours")
        
        print(f"\n[INFO] Training will complete at ~20:00 UTC+1")
        print(f"[INFO] Check logs in: {work_dir}/")
    else:
        print("[WARNING] No checkpoints found yet. Training may be starting...")
    
    # Check for log files
    log_files = list(work_dir.glob('*.log'))
    if log_files:
        print(f"\nLog files found:")
        for log in sorted(log_files):
            size_mb = log.stat().st_size / (1024*1024)
            print(f"  - {log.name} ({size_mb:.1f} MB)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    monitor_training()
