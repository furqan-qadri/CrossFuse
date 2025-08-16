# -*- coding: utf-8 -*-
"""
Temperature Tracking System for Multi-Head Attention Specialization
Tracks and saves temperature data during training for visualization
"""

import torch
import numpy as np
import os
from datetime import datetime

class TemperatureTracker:
    """Simple tracker for multi-head temperature specialization during training"""
    
    def __init__(self, n_heads=16, save_dir="temperature_logs"):
        self.n_heads = n_heads
        self.save_dir = save_dir
        self.epoch_data = []  # Store temperature data for each epoch
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.head_temps_file = os.path.join(save_dir, f"head_temperatures_{timestamp}.txt")
        self.variance_file = os.path.join(save_dir, f"temperature_variance_{timestamp}.txt")
        
        # Write headers
        with open(self.head_temps_file, 'w') as f:
            f.write("# Head Temperature Data for Multi-Head Specialization Analysis\n")
            f.write("# Format: epoch,head_1,head_2,...,head_16\n")
        
        with open(self.variance_file, 'w') as f:
            f.write("# Temperature Variance Data for Specialization Timeline\n")
            f.write("# Format: epoch,variance,mean_temp,min_temp,max_temp\n")
    
    def collect_temperatures(self, model, epoch):
        """Extract temperature values from all cross-attention modules"""
        temperatures = []
        
        # Find all cross-attention modules with temperatures
        for module in model.modules():
            if hasattr(module, 'temperatures') and hasattr(module, 'cross') and module.cross:
                temps = module.temperatures.data.cpu().numpy()
                temperatures.extend(temps)
        
        if len(temperatures) == 0:
            print("Warning: No temperature parameters found in model")
            return
        
        # Take first 16 heads (main cross-attention module)
        temperatures = temperatures[:self.n_heads]
        
        # Store epoch data
        self.epoch_data.append({
            'epoch': epoch,
            'temperatures': temperatures.copy(),
            'variance': np.var(temperatures),
            'mean': np.mean(temperatures),
            'min': np.min(temperatures),
            'max': np.max(temperatures)
        })
        
        # Save to files immediately (in case training stops)
        self._save_current_epoch(epoch, temperatures)
        
        return temperatures
    
    def _save_current_epoch(self, epoch, temperatures):
        """Save current epoch data to files"""
        # Save head temperatures
        with open(self.head_temps_file, 'a') as f:
            temp_str = ','.join([f"{t:.4f}" for t in temperatures])
            f.write(f"{epoch},{temp_str}\n")
        
        # Save variance data
        variance = np.var(temperatures)
        mean_temp = np.mean(temperatures)
        min_temp = np.min(temperatures)
        max_temp = np.max(temperatures)
        
        with open(self.variance_file, 'a') as f:
            f.write(f"{epoch},{variance:.6f},{mean_temp:.4f},{min_temp:.4f},{max_temp:.4f}\n")
    
    def print_specialization_status(self, epoch, temperatures):
        """Print current specialization status"""
        sharp_count = sum(1 for t in temperatures if t < 0.7)
        broad_count = sum(1 for t in temperatures if t > 1.3)
        balanced_count = len(temperatures) - sharp_count - broad_count
        
        print(f"\n🔥 Epoch {epoch} - Temperature Specialization Status:")
        print(f"   Sharp Focus (τ < 0.7):    {sharp_count:2d} heads")
        print(f"   Broad Context (τ > 1.3):  {broad_count:2d} heads") 
        print(f"   Balanced (0.7 ≤ τ ≤ 1.3): {balanced_count:2d} heads")
        print(f"   Variance: {np.var(temperatures):.4f}")
        
        # Show individual head temperatures
        print("   Head Temperatures:")
        for i, temp in enumerate(temperatures):
            category = "SHARP" if temp < 0.7 else "BROAD" if temp > 1.3 else "BALANCED"
            print(f"     Head {i+1:2d}: {temp:.3f} ({category})")
    
    def save_final_summary(self):
        """Save final summary statistics"""
        if not self.epoch_data:
            return
        
        summary_file = os.path.join(self.save_dir, "training_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("# Multi-Head Temperature Specialization Training Summary\n")
            f.write(f"# Total Epochs: {len(self.epoch_data)}\n")
            f.write(f"# Number of Heads: {self.n_heads}\n\n")
            
            # Final epoch statistics
            final_data = self.epoch_data[-1]
            final_temps = final_data['temperatures']
            
            sharp_heads = [i for i, t in enumerate(final_temps) if t < 0.7]
            broad_heads = [i for i, t in enumerate(final_temps) if t > 1.3]
            balanced_heads = [i for i, t in enumerate(final_temps) if 0.7 <= t <= 1.3]
            
            f.write("FINAL SPECIALIZATION RESULTS:\n")
            f.write(f"Sharp Focus Heads (τ < 0.7): {len(sharp_heads)} heads - {sharp_heads}\n")
            f.write(f"Broad Context Heads (τ > 1.3): {len(broad_heads)} heads - {broad_heads}\n")
            f.write(f"Balanced Heads (0.7 ≤ τ ≤ 1.3): {len(balanced_heads)} heads - {balanced_heads}\n\n")
            
            f.write("SPECIALIZATION TIMELINE:\n")
            f.write("Epoch\tVariance\tMean\tMin\tMax\n")
            for data in self.epoch_data:
                f.write(f"{data['epoch']}\t{data['variance']:.4f}\t{data['mean']:.3f}\t{data['min']:.3f}\t{data['max']:.3f}\n")
        
        print(f"\n✅ Temperature tracking complete! Files saved in: {self.save_dir}")
        print(f"   - Head temperatures: {self.head_temps_file}")
        print(f"   - Variance timeline: {self.variance_file}")
        print(f"   - Training summary: {summary_file}")


def integrate_temperature_tracking(model, epoch, tracker=None):
    """
    Simple function to integrate temperature tracking into existing training loop
    
    Args:
        model: The training model
        epoch: Current epoch number
        tracker: TemperatureTracker instance (created if None)
    
    Returns:
        tracker: TemperatureTracker instance for reuse
    """
    if tracker is None:
        tracker = TemperatureTracker()
    
    # Collect and save temperature data
    temperatures = tracker.collect_temperatures(model, epoch)
    
    if temperatures is not None:
        # Print specialization status every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            tracker.print_specialization_status(epoch, temperatures)
    
    return tracker
