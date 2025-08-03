#!/usr/bin/env python3
"""
Test cosine annealing with 16 epochs and create simple LR monitor
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.append('.')
from tools.utils import cosine_annealing_lr

def test_16_epochs():
    """Test cosine schedule with 16 epochs"""
    print("Testing Cosine Annealing with 16 Epochs")
    print("="*50)
    
    # Test parameters matching your testing setup
    base_lr = 0.001  # or whatever you use in args.lr
    total_epochs = 16
    min_lr = 1e-6
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    
    # Track learning rates
    epochs = list(range(total_epochs))
    lrs = []
    
    print(f"Base LR: {base_lr}")
    print(f"Min LR: {min_lr}")
    print(f"Total Epochs: {total_epochs}")
    print("-" * 30)
    
    # Test each epoch
    for epoch in epochs:
        lr = cosine_annealing_lr(optimizer, epoch, base_lr, total_epochs, min_lr)
        lrs.append(lr)
        print(f"Epoch {epoch:2d}: LR = {lr:.6f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: LR schedule (log scale)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, lrs, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing - 16 Epochs (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: LR schedule (linear scale)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, lrs, 'g-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing - 16 Epochs (Linear Scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: LR reduction per epoch
    plt.subplot(2, 2, 3)
    if len(lrs) > 1:
        reductions = [(lrs[i] - lrs[i+1])/lrs[i] * 100 for i in range(len(lrs)-1)]
        plt.plot(epochs[1:], reductions, 'r-o', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('LR Reduction (%)')
        plt.title('Learning Rate Reduction per Epoch')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Compare with other schedules
    plt.subplot(2, 2, 4)
    
    # Step decay every 5 epochs
    step_lrs = [base_lr * (0.5 ** (epoch // 5)) for epoch in epochs]
    
    # Linear decay
    linear_lrs = [base_lr * (1 - epoch / total_epochs) for epoch in epochs]
    
    plt.plot(epochs, lrs, 'b-o', linewidth=2, markersize=3, label='Cosine')
    plt.plot(epochs, step_lrs, 'g-s', linewidth=2, markersize=3, label='Step (every 5)')
    plt.plot(epochs, linear_lrs, 'r-^', linewidth=2, markersize=3, label='Linear')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Schedule Comparison - 16 Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('cosine_16_epochs_test.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: cosine_16_epochs_test.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*40)
    print("16-EPOCH STATISTICS")
    print("="*40)
    print(f"Initial LR:    {lrs[0]:.6f}")
    print(f"Final LR:      {lrs[-1]:.6f}")
    print(f"LR Ratio:      {lrs[0]/lrs[-1]:.1f}x reduction")
    print(f"Mid-point LR:  {lrs[8]:.6f} (epoch 8)")
    
    return lrs

class SimpleLRMonitor:
    """Simple learning rate monitor for 16-epoch training"""
    
    def __init__(self):
        self.epochs = []
        self.lrs = []
        self.losses = []
        
    def log(self, epoch, lr, loss=None):
        """Log learning rate for an epoch"""
        self.epochs.append(epoch)
        self.lrs.append(lr)
        if loss is not None:
            self.losses.append(loss)
        
        # Print every few epochs for short training
        if epoch % 4 == 0 or epoch < 3 or epoch >= 13:
            loss_str = f", Loss={loss:.4f}" if loss else ""
            print(f"Epoch {epoch:2d}: LR={lr:.6f}{loss_str}")
    
    def save_results(self):
        """Save and plot results after training"""
        if not self.epochs:
            return
            
        # Save data
        with open('lr_monitor_16epochs.txt', 'w') as f:
            f.write("Epoch,LearningRate,Loss\n")
            for i in range(len(self.epochs)):
                loss_val = self.losses[i] if i < len(self.losses) else ""
                f.write(f"{self.epochs[i]},{self.lrs[i]:.8f},{loss_val}\n")
        
        # Create simple plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.lrs, 'b-o', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Actual Training LR Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if self.losses:
            plt.subplot(1, 2, 2)
            plt.plot(self.epochs, self.losses, 'r-o', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('actual_training_lr_16epochs.png', dpi=150, bbox_inches='tight')
        print("📊 Training results saved: actual_training_lr_16epochs.png")
        print("📁 Data saved: lr_monitor_16epochs.txt")

def show_integration_code():
    """Show how to add LR monitoring to training scripts"""
    print("\n" + "="*60)
    print("HOW TO ADD LR MONITORING TO YOUR 16-EPOCH TRAINING")
    print("="*60)
    print("\n1. Add this import at the top of your training script:")
    print("from test_cosine_16_epochs import SimpleLRMonitor")
    print("\n2. Before your training loop, add:")
    print("lr_monitor = SimpleLRMonitor()")
    print("\n3. In your training loop, after the cosine_annealing_lr call:")
    print("lr_cur = utils.cosine_annealing_lr(optimizer, e, args.lr, args.epochs, min_lr=1e-6)")
    print("lr_monitor.log(e, lr_cur, loss_all.item())  # Add this line")
    print("\n4. After training completes:")
    print("lr_monitor.save_results()")
    print("\n" + "="*60)

if __name__ == "__main__":
    # Test the 16-epoch schedule
    lrs = test_16_epochs()
    
    # Show integration instructions
    show_integration_code()
    
    # Demo the monitor
    print("\nDemo of SimpleLRMonitor:")
    print("-" * 25)
    monitor = SimpleLRMonitor()
    
    # Simulate 16-epoch training
    for epoch in range(16):
        # Simulate LR from cosine schedule
        lr = 0.001 * (1e-6/0.001 + (1 - 1e-6/0.001) * 0.5 * (1 + np.cos(np.pi * epoch / 16)))
        # Simulate decreasing loss
        fake_loss = 2.0 * np.exp(-epoch/8) + 0.1 * np.random.random()
        monitor.log(epoch, lr, fake_loss)
    
    monitor.save_results()