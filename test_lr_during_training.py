#!/usr/bin/env python3
"""
Monitor learning rate during actual training runs
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class LearningRateMonitor:
    def __init__(self, save_dir="lr_logs"):
        self.lrs = []
        self.epochs = []
        self.losses = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def log(self, epoch, lr, loss=None):
        """Log learning rate and loss for an epoch"""
        self.epochs.append(epoch)
        self.lrs.append(lr)
        if loss is not None:
            self.losses.append(loss)
        
        # Print every 10 epochs or first 5 epochs
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d}: LR={lr:.6f}" + (f", Loss={loss:.4f}" if loss else ""))
    
    def plot_and_save(self, filename_prefix="lr_monitor"):
        """Create plots and save results"""
        if not self.epochs:
            print("No data to plot!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning rate over time
        axes[0, 0].plot(self.epochs, self.lrs, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].set_title('Learning Rate Schedule')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Learning rate (linear scale for first part)
        first_quarter = len(self.epochs) // 4
        if first_quarter > 0:
            axes[0, 1].plot(self.epochs[:first_quarter], self.lrs[:first_quarter], 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate - First Quarter (Linear)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss vs LR (if available)
        if self.losses and len(self.losses) == len(self.lrs):
            axes[1, 0].scatter(self.lrs, self.losses, alpha=0.6, s=20)
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Loss vs Learning Rate')
            axes[1, 0].set_xscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Loss vs Learning Rate (No Data)')
        
        # Plot 4: LR derivative (rate of change)
        if len(self.lrs) > 1:
            lr_changes = np.diff(self.lrs)
            axes[1, 1].plot(self.epochs[1:], lr_changes, 'r-', linewidth=1)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR Change')
            axes[1, 1].set_title('Learning Rate Change per Epoch')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{self.save_dir}/{filename_prefix}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_filename}")
        
        # Save data
        data_filename = f"{self.save_dir}/{filename_prefix}_{timestamp}.txt"
        with open(data_filename, 'w') as f:
            f.write("Epoch,LearningRate" + (",Loss" if self.losses else "") + "\n")
            for i in range(len(self.epochs)):
                line = f"{self.epochs[i]},{self.lrs[i]:.8f}"
                if i < len(self.losses):
                    line += f",{self.losses[i]:.6f}"
                f.write(line + "\n")
        
        print(f"📁 Data saved: {data_filename}")
        
        # Print statistics
        print("\n" + "="*40)
        print("LEARNING RATE STATISTICS")
        print("="*40)
        print(f"Initial LR: {self.lrs[0]:.6f}")
        print(f"Final LR:   {self.lrs[-1]:.6f}")
        print(f"Min LR:     {min(self.lrs):.6f}")
        print(f"Max LR:     {max(self.lrs):.6f}")
        print(f"LR Ratio:   {self.lrs[0]/self.lrs[-1]:.1f}x reduction")
        
        plt.show()
        return plot_filename, data_filename

# Example usage functions to add to your training scripts
def add_lr_monitoring_to_training():
    """
    Example of how to modify your training scripts to monitor LR
    """
    print("To monitor LR in your training, add this code:")
    print("\n" + "="*50)
    print("# At the top of train_conv_trans.py or train_autoencoder.py:")
    print("from test_lr_during_training import LearningRateMonitor")
    print()
    print("# Before the training loop:")
    print("lr_monitor = LearningRateMonitor(save_dir='lr_logs_conv_trans')")
    print()
    print("# In your training loop, after lr_cur calculation:")
    print("lr_cur = utils.cosine_annealing_lr(optimizer, e, args.lr, args.epochs, min_lr=1e-6)")
    print("lr_monitor.log(e, lr_cur, loss_all.item())  # Add this line")
    print()
    print("# After training completes:")
    print("lr_monitor.plot_and_save('cosine_training')")
    print("="*50)

if __name__ == "__main__":
    # Demo with simulated training
    print("Learning Rate Monitor Demo")
    print("="*30)
    
    # Simulate a training run
    import sys
    sys.path.append('.')
    from tools.utils import cosine_annealing_lr
    import torch
    
    # Setup
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    monitor = LearningRateMonitor("demo_logs")
    
    # Simulate training
    total_epochs = 50
    for epoch in range(total_epochs):
        # Get LR using your function
        lr = cosine_annealing_lr(optimizer, epoch, 0.001, total_epochs)
        
        # Simulate some loss (decreasing with noise)
        fake_loss = 1.0 * np.exp(-epoch/20) + 0.1 * np.random.random()
        
        # Log it
        monitor.log(epoch, lr, fake_loss)
    
    # Create plots
    monitor.plot_and_save("demo_cosine")
    
    # Show how to integrate
    add_lr_monitoring_to_training()