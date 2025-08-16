#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Training Dynamics Analysis Visualization
Minimal code for visualizing loss convergence and component contribution
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import msssim

def simulate_training_losses(epochs=50, batches_per_epoch=100):
    """Simulate training loss data for visualization"""
    total_steps = epochs * batches_per_epoch
    
    # Simulate SSIM loss evolution (starts high, decreases)
    ssim_losses = []
    pixel_losses = []
    gradient_losses = []
    
    for step in range(total_steps):
        # SSIM loss: starts at ~0.8, converges to ~0.1
        ssim_base = 0.1 + 0.7 * np.exp(-step / (total_steps * 0.3))
        ssim_noise = np.random.normal(0, 0.02)
        ssim_loss = max(0.05, ssim_base + ssim_noise)
        
        # Pixel loss: starts at ~2.0, converges to ~0.3
        pixel_base = 0.3 + 1.7 * np.exp(-step / (total_steps * 0.4))
        pixel_noise = np.random.normal(0, 0.05)
        pixel_loss = max(0.1, pixel_base + pixel_noise)
        
        # Gradient loss: starts at ~1.5, converges to ~0.2
        grad_base = 0.2 + 1.3 * np.exp(-step / (total_steps * 0.35))
        grad_noise = np.random.normal(0, 0.03)
        grad_loss = max(0.05, grad_base + grad_noise)
        
        ssim_losses.append(ssim_loss)
        pixel_losses.append(pixel_loss)
        gradient_losses.append(grad_loss)
    
    return np.array(ssim_losses), np.array(pixel_losses), np.array(gradient_losses)

def plot_loss_convergence():
    """Plot 1: SSIM loss component evolution during training"""
    ssim_losses, pixel_losses, gradient_losses = simulate_training_losses()
    steps = np.arange(len(ssim_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ssim_losses, 'b-', linewidth=2, label='SSIM Loss', alpha=0.8)
    plt.xlabel('Training Steps')
    plt.ylabel('SSIM Loss Value')
    plt.title('SSIM Loss Convergence During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ssim_loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ssim_losses

def plot_component_contribution():
    """Plot 2: Component contribution analysis with equal weights"""
    ssim_losses, pixel_losses, gradient_losses = simulate_training_losses()
    
    # Apply weights: pixel=1.0, gradient=1.0, ssim=1.0 (equal contribution)
    w_pixel, w_gradient, w_ssim = 1.0, 1.0, 1.0
    
    weighted_pixel = w_pixel * pixel_losses
    weighted_gradient = w_gradient * gradient_losses  
    weighted_ssim = w_ssim * ssim_losses
    
    total_loss = weighted_pixel + weighted_gradient + weighted_ssim
    
    # Calculate contribution percentages
    pixel_contrib = (weighted_pixel / total_loss) * 100
    gradient_contrib = (weighted_gradient / total_loss) * 100
    ssim_contrib = (weighted_ssim / total_loss) * 100
    
    steps = np.arange(len(ssim_losses))
    
    plt.figure(figsize=(12, 6))
    
    # Stacked area plot
    plt.fill_between(steps, 0, pixel_contrib, alpha=0.7, color='red', label=f'Pixel Loss (w={w_pixel})')
    plt.fill_between(steps, pixel_contrib, pixel_contrib + gradient_contrib, alpha=0.7, color='green', label=f'Gradient Loss (w={w_gradient})')
    plt.fill_between(steps, pixel_contrib + gradient_contrib, 100, alpha=0.7, color='blue', label=f'SSIM Loss (w={w_ssim})')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Component Contribution (%)')
    plt.title('Loss Component Contribution Analysis\n(Equal Weights: Pixel=1.0, Gradient=1.0, SSIM=1.0)')
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('component_contribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating Training Dynamics Analysis...")
    
    # Generate both plots
    plot_loss_convergence()
    plot_component_contribution()
    
    print("Plots saved: 'ssim_loss_convergence.png' and 'component_contribution.png'")
