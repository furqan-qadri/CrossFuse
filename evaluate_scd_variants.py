#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCD Variants Evaluation on 21-pairs TNO Dataset
Tests all SCD calculation methods to find which one matches the paper's value of 1.7659
"""

import numpy as np
import cv2
import os
import pandas as pd
from pathlib import Path
from scd_metric import (
    calculate_scd,
    calculate_scd_variant1_scipy,
    calculate_scd_variant2_normalized,
    calculate_scd_variant3_robust,
    calculate_scd_variant4_absolute,
    calculate_scd_variant5_squared,
    calculate_scd_variant6_weighted,
    calculate_scd_variant7_enhanced,
    calculate_scd_variant8_covariance_based,
    calculate_scd_normalized
)

class SCDVariantsEvaluator:
    """
    Evaluates all SCD variants on the 21_pairs_tno dataset
    """
    
    def __init__(self):
        # Dataset paths
        self.ir_dir = "./images/21_pairs_tno/ir"
        self.vis_dir = "./images/21_pairs_tno/vis"
        self.fused_dir = "./output/crossfuse_test/21_pairs_tno_transfuse"
        self.alt_fused_dir = "./test_output/crossfuse_test/21_pairs_tno_transfuse_cpu"
        self.results_dir = "./scd_variants_results"
        
        # Target value from paper
        self.target_scd = 1.7659
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define SCD variants
        self.scd_variants = {
            'Original': calculate_scd,
            'Scipy': calculate_scd_variant1_scipy,
            'Normalized': calculate_scd_variant2_normalized,
            'Robust': calculate_scd_variant3_robust,
            'Absolute': calculate_scd_variant4_absolute,
            'Squared': calculate_scd_variant5_squared,
            'Weighted': calculate_scd_variant6_weighted,
            'Enhanced': calculate_scd_variant7_enhanced,
            'Covariance': calculate_scd_variant8_covariance_based,
            'Normalized_+2': calculate_scd_normalized
        }
    
    def load_image(self, path):
        """Load and preprocess image"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return img.astype(np.float64)
    
    def find_image_files(self):
        """Find all available image files"""
        image_pairs = []
        
        # Check both possible fused directories
        fused_dirs = [self.fused_dir, self.alt_fused_dir]
        
        for fused_dir in fused_dirs:
            if os.path.exists(fused_dir):
                print(f"Using fused directory: {fused_dir}")
                break
        else:
            print("No fused directory found!")
            return []
        
        # Look for image files
        for i in range(1, 22):  # IR1 to IR21
            ir_files = [
                f"{self.ir_dir}/IR{i}.png",
                f"{self.ir_dir}/IR{i}.bmp",
                f"{self.ir_dir}/IR{i}.jpg"
            ]
            
            vis_files = [
                f"{self.vis_dir}/VIS{i}.png",
                f"{self.vis_dir}/VIS{i}.bmp", 
                f"{self.vis_dir}/VIS{i}.jpg"
            ]
            
            fused_files = [
                f"{fused_dir}/results_transfuse_IR{i}.png",
                f"{fused_dir}/results_transfuse_IR{i}.bmp",
                f"{fused_dir}/results_transfuse_IR{i}.jpg"
            ]
            
            # Find existing files
            ir_file = next((f for f in ir_files if os.path.exists(f)), None)
            vis_file = next((f for f in vis_files if os.path.exists(f)), None)
            fused_file = next((f for f in fused_files if os.path.exists(f)), None)
            
            if ir_file and vis_file and fused_file:
                image_pairs.append((ir_file, vis_file, fused_file, f"IR{i}"))
        
        print(f"Found {len(image_pairs)} image pairs")
        return image_pairs
    
    def evaluate_single_image(self, ir_path, vis_path, fused_path, image_name):
        """Evaluate all SCD variants for a single image pair"""
        try:
            # Load images
            ir_img = self.load_image(ir_path)
            vis_img = self.load_image(vis_path)
            fused_img = self.load_image(fused_path)
            
            # Ensure same dimensions
            if not (ir_img.shape == vis_img.shape == fused_img.shape):
                print(f"Warning: Shape mismatch for {image_name}")
                return None
            
            results = {'image': image_name}
            
            # Calculate all SCD variants
            for variant_name, variant_func in self.scd_variants.items():
                try:
                    scd_value = variant_func(ir_img, vis_img, fused_img)
                    results[variant_name] = scd_value
                    results[f'{variant_name}_diff'] = abs(scd_value - self.target_scd)
                except Exception as e:
                    print(f"Error calculating {variant_name} for {image_name}: {e}")
                    results[variant_name] = np.nan
                    results[f'{variant_name}_diff'] = np.inf
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return None
    
    def evaluate_all_images(self):
        """Evaluate all SCD variants on all image pairs"""
        image_pairs = self.find_image_files()
        
        if not image_pairs:
            print("No image pairs found!")
            return None
        
        all_results = []
        
        print(f"Processing {len(image_pairs)} image pairs...")
        for i, (ir_path, vis_path, fused_path, image_name) in enumerate(image_pairs, 1):
            print(f"Processing {i}/{len(image_pairs)}: {image_name}")
            
            result = self.evaluate_single_image(ir_path, vis_path, fused_path, image_name)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("No results obtained!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        summary_stats = self.calculate_summary_statistics(df)
        
        # Save results
        self.save_results(df, summary_stats)
        
        # Display results
        self.display_results(df, summary_stats)
        
        return df, summary_stats
    
    def calculate_summary_statistics(self, df):
        """Calculate summary statistics for all variants"""
        variants = list(self.scd_variants.keys())
        
        summary_stats = {}
        
        for variant in variants:
            if variant in df.columns:
                values = df[variant].dropna()
                diff_values = df[f'{variant}_diff'].dropna()
                
                summary_stats[variant] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'mean_diff_from_target': diff_values.mean(),
                    'min_diff_from_target': diff_values.min(),
                    'count': len(values)
                }
        
        return summary_stats
    
    def save_results(self, df, summary_stats):
        """Save results to files"""
        # Save detailed results
        detailed_file = f"{self.results_dir}/scd_variants_detailed.csv"
        df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats).T
        summary_file = f"{self.results_dir}/scd_variants_summary.csv"
        summary_df.to_csv(summary_file)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Save text report
        report_file = f"{self.results_dir}/scd_variants_report.txt"
        with open(report_file, 'w') as f:
            f.write("SCD Variants Evaluation Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Target SCD value from paper: {self.target_scd}\n\n")
            
            # Ranking by closest to target
            variants = list(self.scd_variants.keys())
            ranking = [(v, summary_stats[v]['mean_diff_from_target']) for v in variants if v in summary_stats]
            ranking.sort(key=lambda x: x[1])
            
            f.write("RANKING (by average difference from target):\n")
            f.write("-" * 50 + "\n")
            for i, (variant, diff) in enumerate(ranking, 1):
                mean_val = summary_stats[variant]['mean']
                f.write(f"{i:2d}. {variant:<15}: {mean_val:8.4f} (avg diff: {diff:6.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED STATISTICS:\n")
            f.write("="*80 + "\n")
            for variant in variants:
                if variant in summary_stats:
                    stats = summary_stats[variant]
                    f.write(f"\n{variant}:\n")
                    f.write(f"  Mean: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                    f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                    f.write(f"  Median: {stats['median']:.6f}\n")
                    f.write(f"  Avg diff from target: {stats['mean_diff_from_target']:.6f}\n")
                    f.write(f"  Min diff from target: {stats['min_diff_from_target']:.6f}\n")
                    f.write(f"  Count: {stats['count']}\n")
        
        print(f"Text report saved to: {report_file}")
    
    def display_results(self, df, summary_stats):
        """Display results summary"""
        print("\n" + "="*80)
        print("SCD VARIANTS EVALUATION RESULTS")
        print("="*80)
        print(f"Target SCD value from paper: {self.target_scd}")
        print(f"Number of image pairs processed: {len(df)}")
        print("-"*80)
        
        # Ranking by closest to target
        variants = list(self.scd_variants.keys())
        ranking = [(v, summary_stats[v]['mean_diff_from_target']) for v in variants if v in summary_stats]
        ranking.sort(key=lambda x: x[1])
        
        print("\nRANKING (by average difference from target):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Variant':<15} {'Mean SCD':<10} {'Avg Diff':<10} {'Min Diff':<10}")
        print("-" * 80)
        
        for i, (variant, avg_diff) in enumerate(ranking, 1):
            stats = summary_stats[variant]
            print(f"{i:2d}.  {variant:<15} {stats['mean']:8.4f}   {avg_diff:8.4f}   {stats['min_diff_from_target']:8.4f}")
        
        print("\n" + "="*80)
        print("RECOMMENDATION:")
        print("="*80)
        best_variant = ranking[0][0]
        best_mean = summary_stats[best_variant]['mean']
        best_diff = ranking[0][1]
        
        print(f"Best variant: {best_variant}")
        print(f"Mean SCD value: {best_mean:.6f}")
        print(f"Average difference from target: {best_diff:.6f}")
        
        if best_diff < 0.1:
            print("✓ This variant is very close to the paper's value!")
        elif best_diff < 0.3:
            print("○ This variant is reasonably close to the paper's value.")
        else:
            print("✗ No variant is very close to the paper's value.")
            print("  Consider checking the dataset, preprocessing, or paper methodology.")

def main():
    print("SCD Variants Evaluation on 21-pairs TNO Dataset")
    print("="*80)
    
    evaluator = SCDVariantsEvaluator()
    
    try:
        df, summary_stats = evaluator.evaluate_all_images()
        
        if df is not None:
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved in: {evaluator.results_dir}/")
        else:
            print("Evaluation failed - no results obtained.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 