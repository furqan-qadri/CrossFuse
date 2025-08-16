#!/usr/bin/env python3
"""
Simple script to visualize learning rate schedules from training logs
"""

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_lr_schedule(lr_log, title, save_path=None):
    """Plot learning rate schedule"""
    epochs = range(1, len(lr_log) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_log, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule: {title}')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LR plot to: {save_path}")
    
    plt.show()

def analyze_training_logs():
    """Analyze all training logs and plot LR schedules"""
    
    # Find all loss data files
    loss_files = glob.glob('models/*/loss/*.mat')
    
    if not loss_files:
        print("No training log files found. Run training first.")
        return
    
    print(f"Found {len(loss_files)} training log files:")
    
    for file_path in loss_files:
        try:
            # Load the .mat file
            data = scio.loadmat(file_path)
            
            if 'lr_log' in data:
                lr_log = data['lr_log'].flatten()
                loss_data = data['loss_data'].flatten()
                
                # Extract model type from filename
                filename = os.path.basename(file_path)
                model_type = filename.split('_')[0]  # 'loss' or similar
                
                print(f"\n📊 {filename}:")
                print(f"   Epochs: {len(lr_log)}")
                print(f"   Initial LR: {lr_log[0]:.6f}")
                print(f"   Final LR: {lr_log[-1]:.6f}")
                print(f"   LR decay: {lr_log[0]/lr_log[-1]:.1f}x")
                
                # Plot LR schedule
                plot_title = f"{model_type} Training - {len(lr_log)} epochs"
                save_path = f"lr_schedule_{model_type}_{len(lr_log)}epochs.png"
                plot_lr_schedule(lr_log, plot_title, save_path)
                
            else:
                print(f"⚠️  {filename}: No LR log found (old format)")
                
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

def test_cosine_implementation():
    """Test the cosine annealing implementation directly"""
    from tools.utils import cosine_annealing_lr
    import torch
    
    # Create dummy optimizer
    dummy_model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    
    # Test parameters
    base_lr = 0.001
    total_epochs = 32
    min_lr = 1e-6
    
    print("🧪 Testing Cosine Annealing Implementation:")
    print(f"   Base LR: {base_lr}")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Min LR: {min_lr}")
    print("\n   Epoch | Learning Rate")
    print("   ------|---------------")
    
    lr_values = []
    for epoch in range(total_epochs):
        lr = cosine_annealing_lr(optimizer, epoch, base_lr, total_epochs, min_lr)
        lr_values.append(lr)
        
        if epoch % 4 == 0:  # Print every 4th epoch
            print(f"   {epoch:5d} | {lr:.6f}")
    
    # Plot test results
    plot_title = f"Cosine Annealing Test: {base_lr} → {min_lr} over {total_epochs} epochs"
    plot_lr_schedule(lr_values, plot_title, "cosine_annealing_test.png")

if __name__ == "__main__":
    print("🔍 Learning Rate Schedule Analyzer")
    print("=" * 40)
    
    # First test the implementation
    test_cosine_implementation()
    
    print("\n" + "=" * 40)
    
    # Then analyze existing logs
    analyze_training_logs() 