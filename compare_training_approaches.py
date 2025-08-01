# -*- coding: utf-8 -*-
# Compare Two-Stage vs End-to-End Training Results
# Run this after training with both approaches

import os
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from datetime import datetime

def load_loss_data(loss_dir, approach_name):
    """Load loss data from .mat files"""
    loss_files = [f for f in os.listdir(loss_dir) if f.endswith('.mat')]
    loss_files.sort()
    
    all_losses = []
    for loss_file in loss_files:
        loss_path = os.path.join(loss_dir, loss_file)
        data = scio.loadmat(loss_path)
        losses = data['loss_data'].flatten()
        all_losses.extend(losses)
    
    return np.array(all_losses)

def compare_approaches():
    """Compare two-stage vs end-to-end training"""
    
    # Paths to loss data
    two_stage_loss_dir = './models/transfuse/loss/'
    end_to_end_loss_dir = './models/end_to_end/loss/'
    
    # Check if both directories exist
    if not os.path.exists(two_stage_loss_dir):
        print(f"❌ Two-stage loss directory not found: {two_stage_loss_dir}")
        print("   Run two-stage training first with train_conv_trans.py")
        return
    
    if not os.path.exists(end_to_end_loss_dir):
        print(f"❌ End-to-end loss directory not found: {end_to_end_loss_dir}")
        print("   Run end-to-end training first with train_end_to_end.py")
        return
    
    # Load loss data
    print("📊 Loading training loss data...")
    try:
        two_stage_losses = load_loss_data(two_stage_loss_dir, "Two-Stage")
        end_to_end_losses = load_loss_data(end_to_end_loss_dir, "End-to-End")
    except Exception as e:
        print(f"❌ Error loading loss data: {e}")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(two_stage_losses, label='Two-Stage', color='blue', alpha=0.7)
    plt.plot(end_to_end_losses, label='End-to-End', color='red', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed loss curves
    plt.subplot(2, 2, 2)
    window_size = min(100, len(two_stage_losses) // 10)
    if window_size > 1:
        two_stage_smooth = np.convolve(two_stage_losses, np.ones(window_size)/window_size, mode='valid')
        end_to_end_smooth = np.convolve(end_to_end_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(two_stage_smooth, label='Two-Stage (Smoothed)', color='blue')
        plt.plot(end_to_end_smooth, label='End-to-End (Smoothed)', color='red')
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothed Loss')
    plt.title('Smoothed Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss distribution
    plt.subplot(2, 2, 3)
    plt.hist(two_stage_losses, bins=50, alpha=0.7, label='Two-Stage', color='blue', density=True)
    plt.hist(end_to_end_losses, bins=50, alpha=0.7, label='End-to-End', color='red', density=True)
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final convergence
    plt.subplot(2, 2, 4)
    final_steps = min(len(two_stage_losses), len(end_to_end_losses), 1000)
    plt.plot(two_stage_losses[-final_steps:], label='Two-Stage (Final)', color='blue')
    plt.plot(end_to_end_losses[-final_steps:], label='End-to-End (Final)', color='red')
    plt.xlabel('Final Training Steps')
    plt.ylabel('Loss')
    plt.title('Final Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'training_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Comparison plot saved: {plot_filename}")
    
    # Print statistics
    print("\n📊 Training Statistics Comparison:")
    print("="*50)
    
    print("Two-Stage Training:")
    print(f"   Total steps: {len(two_stage_losses):,}")
    print(f"   Final loss: {two_stage_losses[-1]:.6f}")
    print(f"   Min loss: {np.min(two_stage_losses):.6f}")
    print(f"   Mean loss: {np.mean(two_stage_losses):.6f}")
    print(f"   Std loss: {np.std(two_stage_losses):.6f}")
    
    print("\nEnd-to-End Training:")
    print(f"   Total steps: {len(end_to_end_losses):,}")
    print(f"   Final loss: {end_to_end_losses[-1]:.6f}")
    print(f"   Min loss: {np.min(end_to_end_losses):.6f}")
    print(f"   Mean loss: {np.mean(end_to_end_losses):.6f}")
    print(f"   Std loss: {np.std(end_to_end_losses):.6f}")
    
    # Performance comparison
    two_stage_final = two_stage_losses[-100:].mean() if len(two_stage_losses) >= 100 else two_stage_losses[-10:].mean()
    end_to_end_final = end_to_end_losses[-100:].mean() if len(end_to_end_losses) >= 100 else end_to_end_losses[-10:].mean()
    
    print(f"\n🏆 Performance Comparison:")
    print("="*50)
    if end_to_end_final < two_stage_final:
        improvement = ((two_stage_final - end_to_end_final) / two_stage_final) * 100
        print(f"✅ End-to-End is BETTER by {improvement:.2f}%")
        print(f"   End-to-End final: {end_to_end_final:.6f}")
        print(f"   Two-Stage final: {two_stage_final:.6f}")
    else:
        degradation = ((end_to_end_final - two_stage_final) / two_stage_final) * 100
        print(f"⚠️  Two-Stage is better by {degradation:.2f}%")
        print(f"   Two-Stage final: {two_stage_final:.6f}")
        print(f"   End-to-End final: {end_to_end_final:.6f}")
    
    # Check model files
    print(f"\n📁 Model Files:")
    print("="*50)
    
    two_stage_models = './models/transfuse/'
    end_to_end_models = './models/end_to_end/'
    
    if os.path.exists(two_stage_models):
        two_stage_files = [f for f in os.listdir(two_stage_models) if f.endswith('.model')]
        print(f"Two-Stage models: {len(two_stage_files)} files")
        if two_stage_files:
            print(f"   Latest: {sorted(two_stage_files)[-1]}")
    
    if os.path.exists(end_to_end_models):
        end_to_end_files = [f for f in os.listdir(end_to_end_models) if f.endswith('.model')]
        print(f"End-to-End models: {len(end_to_end_files)} files")
        if end_to_end_files:
            fusion_models = [f for f in end_to_end_files if 'fusion' in f]
            ir_models = [f for f in end_to_end_files if 'ir_encoder' in f]
            vi_models = [f for f in end_to_end_files if 'vi_encoder' in f]
            print(f"   Fusion models: {len(fusion_models)}")
            print(f"   IR encoder models: {len(ir_models)}")
            print(f"   VIS encoder models: {len(vi_models)}")
    
    plt.show()

if __name__ == "__main__":
    print("🔄 CrossFuse Training Approach Comparison")
    print("=" * 50)
    compare_approaches()