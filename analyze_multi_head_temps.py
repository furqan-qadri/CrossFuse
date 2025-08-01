# -*- coding: utf-8 -*-
# Multi-Head Temperature Specialization Analysis Script
# Run this after training to analyze learned temperature patterns

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def analyze_head_specialization(model_path, args_config=None):
    """
    Analyze learned temperature specialization patterns
    
    Args:
        model_path: Path to trained model (.model file)
        args_config: Optional config for model initialization
    """
    
    print("🔥 Multi-Head Temperature Specialization Analysis")
    print("=" * 60)
    
    # Load model
    try:
        from network.net_conv_trans import Trans_FuseNet
        
        # Model configuration (adjust based on your setup)
        custom_config = {
            "img_size": 32,
            "patch_size": 2,
            "en_out_channels1": 32,
            "out_channels": 1,
            "part_out": 128,
            "train_flag": True,
            "depth_self": 1,
            "depth_cross": 1,
            "n_heads": 16,
            "qkv_bias": True,
            "mlp_ratio": 4,
            "p": 0.,
            "attn_p": 0.,
        }
        
        model = Trans_FuseNet(**custom_config)
        
        # Load trained weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ Model loaded from: {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Extract temperatures from all cross-attention modules
    all_temps = []
    module_info = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'temperatures') and hasattr(module, 'cross') and module.cross:
            temps = module.temperatures.data.cpu().numpy()
            all_temps.extend(temps)
            module_info.append({
                'name': name,
                'temperatures': temps,
                'n_heads': len(temps)
            })
            print(f"📍 Found multi-head temperatures in: {name} ({len(temps)} heads)")
    
    if not all_temps:
        print("❌ No multi-head temperatures found in model!")
        print("   Make sure the model was trained with multi-head temperature specialization.")
        return
    
    all_temps = np.array(all_temps)
    
    # Analysis
    print(f"\n📊 Temperature Specialization Analysis:")
    print(f"   Total attention heads: {len(all_temps)}")
    print(f"   Temperature range: {np.min(all_temps):.3f} - {np.max(all_temps):.3f}")
    print(f"   Average temperature: {np.mean(all_temps):.3f}")
    print(f"   Standard deviation: {np.std(all_temps):.3f}")
    
    # Categorize heads
    sharp_heads = np.sum(all_temps < 0.7)
    broad_heads = np.sum(all_temps > 1.3)
    balanced_heads = len(all_temps) - sharp_heads - broad_heads
    
    print(f"\n🎯 Head Specialization Categories:")
    print(f"   Sharp specialists (< 0.7):     {sharp_heads:2d} heads ({sharp_heads/len(all_temps)*100:.1f}%)")
    print(f"   Broad specialists (> 1.3):     {broad_heads:2d} heads ({broad_heads/len(all_temps)*100:.1f}%)")
    print(f"   Balanced fusion (0.7-1.3):     {balanced_heads:2d} heads ({balanced_heads/len(all_temps)*100:.1f}%)")
    
    # Detailed head analysis
    print(f"\n🔍 Individual Head Analysis:")
    for i, temp in enumerate(all_temps):
        if temp < 0.5:
            specialization = "🔥 VERY SHARP - Edge/Detail specialist"
        elif temp < 0.7:
            specialization = "⚡ SHARP - Feature specialist"
        elif temp > 2.0:
            specialization = "🌐 VERY BROAD - Global context"
        elif temp > 1.3:
            specialization = "📡 BROAD - Contextual specialist"
        else:
            specialization = "⚖️  BALANCED - General fusion"
        
        print(f"   Head {i:2d}: temp={temp:.3f} - {specialization}")
    
    # Visualization
    create_visualizations(all_temps, module_info, model_path)
    
    # Save analysis results
    save_analysis_results(all_temps, module_info, model_path)
    
    print(f"\n✅ Analysis complete! Check generated plots and analysis files.")

def create_visualizations(all_temps, module_info, model_path):
    """Create visualization plots"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Temperature specialization bar chart
    plt.figure(figsize=(15, 10))
    
    # Main temperature plot
    plt.subplot(2, 2, 1)
    colors = ['red' if t < 0.7 else 'green' if t > 1.3 else 'blue' for t in all_temps]
    bars = plt.bar(range(len(all_temps)), all_temps, color=colors, alpha=0.7)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.8, label='Sharp Threshold (0.7)')
    plt.axhline(y=1.3, color='green', linestyle='--', alpha=0.8, label='Broad Threshold (1.3)')
    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Original Fixed (1.0)')
    plt.xlabel('Attention Head Index')
    plt.ylabel('Learned Temperature')
    plt.title('Multi-Head Temperature Specialization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Temperature distribution histogram
    plt.subplot(2, 2, 2)
    plt.hist(all_temps, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.8, label='Sharp Threshold')
    plt.axvline(x=1.3, color='green', linestyle='--', alpha=0.8, label='Broad Threshold')
    plt.axvline(x=1.0, color='black', linestyle='-', alpha=0.8, label='Original Fixed')
    plt.xlabel('Temperature Value')
    plt.ylabel('Number of Heads')
    plt.title('Temperature Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Specialization pie chart
    plt.subplot(2, 2, 3)
    sharp_count = np.sum(all_temps < 0.7)
    broad_count = np.sum(all_temps > 1.3)
    balanced_count = len(all_temps) - sharp_count - broad_count
    
    sizes = [sharp_count, balanced_count, broad_count]
    labels = ['Sharp\n(< 0.7)', 'Balanced\n(0.7-1.3)', 'Broad\n(> 1.3)']
    colors = ['red', 'blue', 'green']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, alpha=0.7)
    plt.title('Head Specialization Distribution')
    
    # Plot 4: Temperature evolution comparison
    plt.subplot(2, 2, 4)
    sorted_temps = np.sort(all_temps)
    plt.plot(range(len(sorted_temps)), sorted_temps, 'o-', linewidth=2, markersize=4)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.8, label='Sharp Threshold')
    plt.axhline(y=1.3, color='green', linestyle='--', alpha=0.8, label='Broad Threshold')
    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.8, label='Original Fixed')
    plt.xlabel('Head Rank (sorted by temperature)')
    plt.ylabel('Temperature Value')
    plt.title('Temperature Specialization Range')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'multi_head_temperature_analysis_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_filename}")
    
    plt.show()

def save_analysis_results(all_temps, module_info, model_path):
    """Save detailed analysis results to file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'multi_head_analysis_{timestamp}.txt'
    
    with open(results_filename, 'w') as f:
        f.write("Multi-Head Temperature Specialization Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Temperature Statistics:\n")
        f.write(f"  Total heads: {len(all_temps)}\n")
        f.write(f"  Min temperature: {np.min(all_temps):.6f}\n")
        f.write(f"  Max temperature: {np.max(all_temps):.6f}\n")
        f.write(f"  Mean temperature: {np.mean(all_temps):.6f}\n")
        f.write(f"  Std deviation: {np.std(all_temps):.6f}\n\n")
        
        sharp_count = np.sum(all_temps < 0.7)
        broad_count = np.sum(all_temps > 1.3)
        balanced_count = len(all_temps) - sharp_count - broad_count
        
        f.write(f"Specialization Categories:\n")
        f.write(f"  Sharp specialists (< 0.7): {sharp_count} heads ({sharp_count/len(all_temps)*100:.1f}%)\n")
        f.write(f"  Broad specialists (> 1.3): {broad_count} heads ({broad_count/len(all_temps)*100:.1f}%)\n")
        f.write(f"  Balanced fusion (0.7-1.3): {balanced_count} heads ({balanced_count/len(all_temps)*100:.1f}%)\n\n")
        
        f.write(f"Individual Head Analysis:\n")
        for i, temp in enumerate(all_temps):
            if temp < 0.5:
                specialization = "VERY SHARP - Edge/Detail specialist"
            elif temp < 0.7:
                specialization = "SHARP - Feature specialist"
            elif temp > 2.0:
                specialization = "VERY BROAD - Global context"
            elif temp > 1.3:
                specialization = "BROAD - Contextual specialist"
            else:
                specialization = "BALANCED - General fusion"
            
            f.write(f"  Head {i:2d}: temp={temp:.6f} - {specialization}\n")
        
        f.write(f"\nModule Details:\n")
        for info in module_info:
            f.write(f"  {info['name']}: {info['n_heads']} heads\n")
            for i, temp in enumerate(info['temperatures']):
                f.write(f"    Head {i}: {temp:.6f}\n")
    
    print(f"📄 Detailed analysis saved: {results_filename}")

def analyze_latest_model():
    """Convenience function to analyze the latest trained model"""
    
    # Check for end-to-end models first
    end_to_end_dir = "./models/end_to_end/"
    two_stage_dir = "./models/transfuse/"
    
    latest_model = None
    
    # Look for end-to-end models
    if os.path.exists(end_to_end_dir):
        models = [f for f in os.listdir(end_to_end_dir) if f.endswith('.model') and 'fusion' in f]
        if models:
            latest_model = os.path.join(end_to_end_dir, sorted(models)[-1])
    
    # Fall back to two-stage models
    if not latest_model and os.path.exists(two_stage_dir):
        models = [f for f in os.listdir(two_stage_dir) if f.endswith('.model')]
        if models:
            latest_model = os.path.join(two_stage_dir, sorted(models)[-1])
    
    if latest_model:
        print(f"🔍 Analyzing latest model: {latest_model}")
        analyze_head_specialization(latest_model)
    else:
        print("❌ No trained models found!")
        print("Available directories:")
        print(f"  {end_to_end_dir}: {'✅' if os.path.exists(end_to_end_dir) else '❌'}")
        print(f"  {two_stage_dir}: {'✅' if os.path.exists(two_stage_dir) else '❌'}")

if __name__ == "__main__":
    print("🔥 Multi-Head Temperature Specialization Analyzer")
    print("=" * 60)
    
    # Option 1: Analyze specific model
    # analyze_head_specialization("./models/end_to_end/end_to_end_fusion_epoch_32.model")
    
    # Option 2: Analyze latest model automatically
    analyze_latest_model()