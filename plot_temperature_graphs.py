# -*- coding: utf-8 -*-
"""
Simple plotting script for temperature specialization graphs
Reads data from temperature tracking files and creates the two required graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

def load_temperature_data(log_dir="temperature_logs"):
    """Load temperature data from log files"""
    
    # Find the most recent temperature files
    head_temp_files = glob.glob(os.path.join(log_dir, "head_temperatures_*.txt"))
    variance_files = glob.glob(os.path.join(log_dir, "temperature_variance_*.txt"))
    
    if not head_temp_files or not variance_files:
        raise FileNotFoundError(f"No temperature log files found in {log_dir}")
    
    # Use the most recent files
    head_temp_file = max(head_temp_files, key=os.path.getmtime)
    variance_file = max(variance_files, key=os.path.getmtime)
    
    print(f"Loading data from:")
    print(f"  - Head temperatures: {head_temp_file}")
    print(f"  - Variance data: {variance_file}")
    
    # Load head temperatures
    head_data = []
    with open(head_temp_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) > 1:
                epoch = int(parts[0])
                temperatures = [float(x) for x in parts[1:]]
                head_data.append((epoch, temperatures))
    
    # Load variance data
    variance_data = []
    with open(variance_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                epoch = int(parts[0])
                variance = float(parts[1])
                variance_data.append((epoch, variance))
    
    return head_data, variance_data

def create_head_temperature_bar_chart(head_data, save_path="temperature_graphs"):
    """Create Graph 1: Head Temperature Bar Chart"""
    
    if not head_data:
        print("No head temperature data available")
        return
    
    # Use final epoch data
    final_epoch, final_temperatures = head_data[-1]
    
    # Ensure we have exactly 16 heads
    temperatures = final_temperatures[:16]
    heads = list(range(1, len(temperatures) + 1))
    
    # Color-code bars based on temperature ranges
    colors = []
    for temp in temperatures:
        if temp < 0.7:
            colors.append('red')      # Sharp focus
        elif temp > 1.3:
            colors.append('blue')     # Global context  
        else:
            colors.append('green')    # Balanced
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(heads, temperatures, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Attention Head Number', fontsize=12)
    plt.ylabel('Temperature Value (τ)', fontsize=12)
    plt.title(f'Multi-Head Temperature Specialization (Final - Epoch {final_epoch})', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal reference lines
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Sharp Focus Threshold (τ < 0.7)')
    plt.axhline(y=1.3, color='blue', linestyle='--', alpha=0.5, label='Global Context Threshold (τ > 1.3)')
    
    # Add value labels on bars
    for bar, temp in zip(bars, temperatures):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{temp:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.8, label='Sharp Focus (τ < 0.7)')
    green_patch = mpatches.Patch(color='green', alpha=0.8, label='Balanced (0.7 ≤ τ ≤ 1.3)')
    blue_patch = mpatches.Patch(color='blue', alpha=0.8, label='Global Context (τ > 1.3)')
    plt.legend(handles=[red_patch, green_patch, blue_patch], loc='upper right')
    
    plt.xticks(heads)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"head_temperature_bar_chart_epoch_{final_epoch}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph 1 saved: {filename}")
    
    plt.show()

def create_specialization_timeline(variance_data, save_path="temperature_graphs"):
    """Create Graph 2: Specialization Timeline"""
    
    if not variance_data:
        print("No variance data available")
        return
    
    epochs = [data[0] for data in variance_data]
    variances = [data[1] for data in variance_data]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, variances, marker='o', linewidth=2, markersize=4, color='darkblue')
    
    # Add horizontal line for final variance (stable specialization)
    if len(variances) > 0:
        final_variance = variances[-1]
        plt.axhline(y=final_variance, color='red', linestyle='--', alpha=0.7, 
                   label=f'Final Specialization (σ² = {final_variance:.4f})')
    
    # Customize the plot
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Temperature Variance (σ²)', fontsize=12)
    plt.title('Multi-Head Temperature Specialization Timeline', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for key phases
    if len(variances) >= 3:
        # Find approximate specialization start (when variance starts increasing significantly)
        variance_diff = np.diff(variances)
        if len(variance_diff) > 0:
            max_increase_idx = np.argmax(variance_diff) + 1
            if max_increase_idx < len(epochs):
                plt.annotate('Specialization Begins', 
                           xy=(epochs[max_increase_idx], variances[max_increase_idx]),
                           xytext=(epochs[max_increase_idx] + 2, variances[max_increase_idx] + 0.01),
                           arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                           fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "specialization_timeline.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph 2 saved: {filename}")
    
    plt.show()

def main():
    """Main function to create both graphs"""
    
    print("🔥 Creating Multi-Head Temperature Specialization Graphs...")
    
    try:
        # Load data
        head_data, variance_data = load_temperature_data()
        
        print(f"Loaded data for {len(head_data)} epochs")
        
        # Create both graphs
        create_head_temperature_bar_chart(head_data)
        create_specialization_timeline(variance_data)
        
        # Print summary statistics
        if head_data:
            final_epoch, final_temps = head_data[-1]
            sharp_count = sum(1 for t in final_temps if t < 0.7)
            broad_count = sum(1 for t in final_temps if t > 1.3)
            balanced_count = len(final_temps) - sharp_count - broad_count
            
            print(f"\n📊 Final Specialization Summary (Epoch {final_epoch}):")
            print(f"   Sharp Focus Heads: {sharp_count}")
            print(f"   Broad Context Heads: {broad_count}")
            print(f"   Balanced Heads: {balanced_count}")
            print(f"   Temperature Variance: {np.var(final_temps):.4f}")
        
        print("\n✅ Graphs created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating graphs: {e}")
        print("Make sure you have run training with temperature tracking first.")

if __name__ == "__main__":
    main()
