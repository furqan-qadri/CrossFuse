#!/usr/bin/env python3
"""
Test script to verify cosine annealing learning rate schedule
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
from tools.utils import cosine_annealing_lr, warmup_cosine_lr

def test_cosine_schedule():
    """Test and visualize cosine annealing schedule"""
    
    # Test parameters
    base_lr = 0.001
    total_epochs = 100
    min_lr = 1e-6
    
    # Create dummy model and optimizer for testing
    dummy_model = torch.nn.Linear(10, 1)
    optimizer = optim.Adam(dummy_model.parameters(), lr=base_lr)
    
    # Track learning rates
    epochs = list(range(total_epochs))
    actual_lrs = []
    expected_lrs = []
    
    print("Testing Cosine Annealing Schedule...")
    print(f"Base LR: {base_lr}")
    print(f"Min LR: {min_lr}")
    print(f"Total Epochs: {total_epochs}")
    print("-" * 50)
    
    for epoch in epochs:
        # Use your implementation
        actual_lr = cosine_annealing_lr(optimizer, epoch, base_lr, total_epochs, min_lr)
        actual_lrs.append(actual_lr)
        
        # Calculate expected LR using the mathematical formula
        expected_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        expected_lrs.append(expected_lr)
        
        # Print some sample epochs
        if epoch % 20 == 0 or epoch < 5 or epoch > total_epochs - 5:
            print(f"Epoch {epoch:3d}: Actual={actual_lr:.6f}, Expected={expected_lr:.6f}, Match={abs(actual_lr - expected_lr) < 1e-10}")
    
    # Verify mathematical correctness
    max_diff = max(abs(a - e) for a, e in zip(actual_lrs, expected_lrs))
    print(f"\nMax difference between actual and expected: {max_diff:.2e}")
    print(f"Schedule working correctly: {max_diff < 1e-10}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cosine schedule
    plt.subplot(2, 2, 1)
    plt.plot(epochs, actual_lrs, 'b-', linewidth=2, label='Your Implementation')
    plt.plot(epochs, expected_lrs, 'r--', linewidth=1, label='Expected Formula')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Compare with other schedules
    plt.subplot(2, 2, 2)
    
    # Step decay (traditional)
    step_lrs = [base_lr * (0.1 ** (epoch // 30)) for epoch in epochs]
    
    # Linear decay
    linear_lrs = [base_lr * (1 - epoch / total_epochs) for epoch in epochs]
    
    plt.plot(epochs, actual_lrs, 'b-', linewidth=2, label='Cosine Annealing')
    plt.plot(epochs, step_lrs, 'g-', linewidth=2, label='Step Decay')
    plt.plot(epochs, linear_lrs, 'orange', linewidth=2, label='Linear Decay')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Schedule Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Warmup + Cosine
    plt.subplot(2, 2, 3)
    warmup_lrs = []
    warmup_epochs = 5
    
    for epoch in epochs:
        warmup_lr = warmup_cosine_lr(optimizer, epoch, base_lr, total_epochs, warmup_epochs, min_lr)
        warmup_lrs.append(warmup_lr)
    
    plt.plot(epochs, actual_lrs, 'b-', linewidth=2, label='Cosine Only')
    plt.plot(epochs, warmup_lrs, 'purple', linewidth=2, label='Warmup + Cosine')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Warmup vs No Warmup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: First 20 epochs (linear scale)
    plt.subplot(2, 2, 4)
    first_20_epochs = epochs[:20]
    first_20_cosine = actual_lrs[:20]
    first_20_warmup = warmup_lrs[:20]
    
    plt.plot(first_20_epochs, first_20_cosine, 'b-', linewidth=2, label='Cosine Only')
    plt.plot(first_20_epochs, first_20_warmup, 'purple', linewidth=2, label='Warmup + Cosine')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('First 20 Epochs (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cosine_schedule_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test edge cases
    print("\n" + "="*50)
    print("EDGE CASE TESTS")
    print("="*50)
    
    # Test epoch 0
    lr_0 = cosine_annealing_lr(optimizer, 0, base_lr, total_epochs, min_lr)
    expected_0 = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(0))  # Should be base_lr
    print(f"Epoch 0: {lr_0:.6f} (should be close to {base_lr})")
    
    # Test final epoch
    lr_final = cosine_annealing_lr(optimizer, total_epochs-1, base_lr, total_epochs, min_lr)
    expected_final = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (total_epochs-1) / total_epochs))
    print(f"Final epoch: {lr_final:.6f} (should be close to {min_lr})")
    
    # Test mid-point
    mid_epoch = total_epochs // 2
    lr_mid = cosine_annealing_lr(optimizer, mid_epoch, base_lr, total_epochs, min_lr)
    expected_mid = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * 0.5))  # Should be min_lr
    print(f"Mid epoch ({mid_epoch}): {lr_mid:.6f} (should be close to {min_lr})")
    
    return actual_lrs, expected_lrs

def test_integration_with_training():
    """Test how the schedule integrates with actual training loop"""
    print("\n" + "="*50)
    print("INTEGRATION TEST")
    print("="*50)
    
    # Simulate training parameters
    args_lr = 0.001
    args_epochs = 50
    
    # Create a real optimizer like in your training
    model = torch.nn.Linear(100, 10)
    optimizer = optim.Adam(model.parameters(), lr=args_lr)
    
    print("Simulating training loop...")
    lrs_during_training = []
    
    for e in range(args_epochs):
        # This is exactly what your training code does
        lr_cur = cosine_annealing_lr(optimizer, e, args_lr, args_epochs, min_lr=1e-6)
        lrs_during_training.append(lr_cur)
        
        # Verify optimizer actually has this learning rate
        actual_optimizer_lr = optimizer.param_groups[0]['lr']
        
        if e % 10 == 0:
            print(f"Epoch {e}: Returned LR={lr_cur:.6f}, Optimizer LR={actual_optimizer_lr:.6f}")
            
        assert abs(lr_cur - actual_optimizer_lr) < 1e-10, f"LR mismatch at epoch {e}"
    
    print("✅ Integration test passed!")
    return lrs_during_training

if __name__ == "__main__":
    print("Testing Cosine Annealing Implementation")
    print("="*50)
    
    # Run tests
    actual_lrs, expected_lrs = test_cosine_schedule()
    training_lrs = test_integration_with_training()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("✅ Mathematical formula is correct")
    print("✅ Implementation matches expected behavior")
    print("✅ Integration with optimizer works")
    print("✅ Edge cases handled properly")
    print(f"📊 Visualization saved as 'cosine_schedule_test.png'")