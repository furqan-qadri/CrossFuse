#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Training Dynamics Tracker for Real-time Loss Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

class TrainingDynamicsTracker:
    def __init__(self, save_path="./plots", model_name="model"):
        self.save_path = save_path
        self.model_name = model_name
        self.loss_history = defaultdict(list)
        self.step_count = 0
        
        # Create save directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def update_losses(self, **losses):
        """Update loss history with current step losses"""
        self.step_count += 1
        for loss_name, loss_value in losses.items():
            # Convert tensor to float if needed
            if hasattr(loss_value, 'item'):
                loss_value = loss_value.item()
            self.loss_history[loss_name].append(loss_value)
    
    def plot_ssim_convergence(self, save=True):
        """Plot SSIM loss convergence"""
        if 'ssim_loss' not in self.loss_history:
            print("No SSIM loss data available for plotting")
            return
            
        plt.figure(figsize=(10, 6))
        steps = np.arange(len(self.loss_history['ssim_loss']))
        plt.plot(steps, self.loss_history['ssim_loss'], 'b-', linewidth=2, label='SSIM Loss', alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('SSIM Loss Value')
        plt.title(f'{self.model_name} - SSIM Loss Convergence During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_path, f'{self.model_name}_ssim_convergence.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SSIM convergence plot saved: {save_path}")
        plt.show()
    
    def plot_component_contribution(self, component_weights=None, save=True):
        """Plot loss component contributions"""
        # Default weights based on model type
        if component_weights is None:
            if 'pix_loss' in self.loss_history:  # TransFuse model
                component_weights = {'pix_loss': 1.0, 'gra_loss': 1.0, 'ssim_loss': 1.0}
            else:  # Autoencoder model
                component_weights = {'recon_loss': 1.0, 'ssim_loss': 1.0}
        
        # Filter available loss components
        available_components = {k: v for k, v in component_weights.items() if k in self.loss_history}
        
        if not available_components:
            print("No matching loss components found for contribution analysis")
            return
        
        # Calculate weighted losses
        weighted_losses = {}
        for comp_name, weight in available_components.items():
            weighted_losses[comp_name] = np.array(self.loss_history[comp_name]) * weight
        
        # Calculate total loss for each step
        total_losses = np.sum(list(weighted_losses.values()), axis=0)
        
        # Calculate contribution percentages
        contributions = {}
        for comp_name, weighted_loss in weighted_losses.items():
            contributions[comp_name] = (weighted_loss / total_losses) * 100
        
        # Plot stacked area chart
        plt.figure(figsize=(12, 6))
        steps = np.arange(len(total_losses))
        
        # Colors for different components
        colors = {'pix_loss': 'red', 'recon_loss': 'red', 'gra_loss': 'green', 'ssim_loss': 'blue'}
        labels = {'pix_loss': 'Pixel Loss', 'recon_loss': 'Reconstruction Loss', 'gra_loss': 'Gradient Loss', 'ssim_loss': 'SSIM Loss'}
        
        bottom = np.zeros(len(steps))
        for comp_name, contrib in contributions.items():
            color = colors.get(comp_name, 'gray')
            label = f"{labels.get(comp_name, comp_name)} (w={available_components[comp_name]})"
            plt.fill_between(steps, bottom, bottom + contrib, alpha=0.7, color=color, label=label)
            bottom += contrib
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Component Contribution (%)')
        plt.title(f'{self.model_name} - Loss Component Contribution Analysis')
        plt.legend(loc='center right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_path, f'{self.model_name}_component_contribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Component contribution plot saved: {save_path}")
        plt.show()
    
    def plot_all_losses(self, save=True):
        """Plot all individual loss components"""
        if not self.loss_history:
            print("No loss data available for plotting")
            return
        
        plt.figure(figsize=(15, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (loss_name, loss_values) in enumerate(self.loss_history.items()):
            if loss_name != 'total_loss':  # Plot total loss separately
                steps = np.arange(len(loss_values))
                color = colors[i % len(colors)]
                plt.plot(steps, loss_values, color=color, linewidth=2, label=loss_name, alpha=0.8)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value')
        plt.title(f'{self.model_name} - Individual Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_path, f'{self.model_name}_all_losses.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"All losses plot saved: {save_path}")
        plt.show()
    
    def generate_training_report(self, save=True):
        """Generate comprehensive training dynamics analysis"""
        print(f"\n=== {self.model_name} Training Dynamics Analysis ===")
        
        # Generate all plots
        self.plot_ssim_convergence(save=save)
        self.plot_component_contribution(save=save)
        self.plot_all_losses(save=save)
        
        print(f"Training dynamics analysis completed for {self.model_name}")
        print(f"Total training steps: {self.step_count}")
        print(f"Plots saved to: {self.save_path}")
