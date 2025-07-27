#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Evaluation Script for 3_pairs_tno Dataset (Testing) - SIMPLIFIED VERSION
Evaluates first 3 image pairs using 2 core fusion metrics: Entropy (EN) and Standard Deviation (SD)
Note: SD is calculated on histogram-equalized images to match CrossFuse paper methodology
"""

import numpy as np
import cv2
from scipy.stats import entropy
import os
import pandas as pd
from pathlib import Path

class BatchFusionEvaluator:
    """
    Evaluates fusion quality on the first 3 pairs of 21_pairs_tno dataset
    """
    
    def __init__(self):
        # Dataset paths
        self.ir_dir = "./images/21_pairs_tno/ir"
        self.vis_dir = "./images/21_pairs_tno/vis"
        self.fused_dir = "./output/crossfuse_test/21_pairs_tno_transfuse"
        self.results_dir = "./evaluation_results_21pairs"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_image(self, path, normalize_to_01=False):
        """Load and preprocess image"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        
        img = img.astype(np.float64)
        
        if normalize_to_01:
            img = img / 255.0  # Normalize to [0, 1]
        
        return img
    
    def calculate_entropy(self, img):
        """Metric 1: Entropy (EN) ↑
        Formula: Standard Shannon entropy based on histogram
        Purpose: Measures information content/richness in fused image
        """
        img_uint = img.astype(np.uint8)
        hist, _ = np.histogram(img_uint, bins=256, range=[0, 256])
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        return entropy(prob, base=2)
    
    def calculate_standard_deviation(self, img):
        """Metric 2: Standard Deviation (SD) ↑
        Formula: SD after histogram equalization (matches CrossFuse paper methodology)
        Purpose: Measures contrast and detail preservation
        """
        img_eq = cv2.equalizeHist(img.astype(np.uint8))
        return np.std(img_eq.astype(np.float64))
    
    def evaluate_single_pair(self, pair_id):
        """Evaluate a single image pair"""
        # Construct file paths
        ir_path = os.path.join(self.ir_dir, f"IR{pair_id}.png")
        vis_path = os.path.join(self.vis_dir, f"VIS{pair_id}.png")
        fused_path = os.path.join(self.fused_dir, f"results_transfuse_IR{pair_id}.png")
        
        # Check if all files exist
        if not all(os.path.exists(p) for p in [ir_path, vis_path, fused_path]):
            print(f"❌ Missing files for pair {pair_id}")
            return None
        
        try:
            # Load images (use [0,255] range for all metrics)
            ir_img = self.load_image(ir_path, normalize_to_01=False)
            vis_img = self.load_image(vis_path, normalize_to_01=False)
            fused_img = self.load_image(fused_path, normalize_to_01=False)
            
        except Exception as e:
            print(f"❌ Error loading images for pair {pair_id}: {e}")
            return None
        
        # Calculate core metrics that match the paper
        results = {
            'pair_id': int(pair_id),
            'EN': self.calculate_entropy(fused_img),
            'SD': self.calculate_standard_deviation(fused_img)
        }
        
        return results
    
    def evaluate_all_pairs(self):
        """Evaluate first 3 image pairs (testing phase)"""
        print("🚀 Starting batch evaluation of first 3 pairs from 21_pairs_tno dataset...")
        print("="*70)
        
        all_results = []
        failed_pairs = []
        
        # Process first 3 pairs only
        for pair_id in range(1, 4):  # 1 to 3
            print(f"📊 Processing pair {pair_id}/3...")
            
            result = self.evaluate_single_pair(pair_id)
            
            if result is not None:
                all_results.append(result)
                print(f"   ✅ Pair {pair_id} completed")
                # Print individual results  
                print(f"      EN: {result['EN']:.4f}")
                print(f"      SD: {result['SD']:.4f}")
            else:
                failed_pairs.append(pair_id)
                print(f"   ❌ Pair {pair_id} failed")
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(all_results)
        
        if len(df) > 0:
            # Ensure pair_id column is integer type
            df['pair_id'] = df['pair_id'].astype(int)
            
            # Calculate statistics
            self.print_summary_results(df, failed_pairs)
            
            # Save detailed results
            self.save_results(df)
        else:
            print("❌ No pairs were successfully evaluated!")
        
        return df
    
    def print_summary_results(self, df, failed_pairs):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("📊 BATCH EVALUATION RESULTS - First 3 pairs from 21_pairs_tno DATASET")
        print("="*80)
        
        if failed_pairs:
            print(f"⚠️  Failed pairs: {failed_pairs}")
        
        print(f"✅ Successfully evaluated: {len(df)}/3 pairs")
        print("-"*80)
        
        # Summary statistics
        metrics = ['EN', 'SD']
        
        print("SUMMARY STATISTICS (Higher values = Better performance):")
        print("-"*60)
        print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-"*60)
        
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            print(f"{metric:<12} {mean_val:<8.4f} {std_val:<8.4f} {min_val:<8.4f} {max_val:<8.4f}")
        
        print("="*80)
        
        # Compare with CrossFuse paper results
        print("\n🎯 COMPARISON WITH CROSSFUSE PAPER RESULTS:")
        print("-"*60)
        crossfuse_results = {
            'EN': 6.8389,
            'SD': 73.4712
        }
        
        print(f"{'Metric':<12} {'Our Mean':<10} {'Paper':<10} {'Difference':<12} {'Error %':<10}")
        print("-"*60)
        
        for metric in metrics:
            our_mean = df[metric].mean()
            paper_val = crossfuse_results[metric]
            diff = our_mean - paper_val
            error_pct = abs(diff) / paper_val * 100
            print(f"{metric:<12} {our_mean:<10.4f} {paper_val:<10.4f} {diff:<+12.4f} {error_pct:<10.1f}%")
        
        print("="*60)
    
    def save_results(self, df):
        """Save results to files"""
        # Save detailed results
        csv_path = os.path.join(self.results_dir, "detailed_results_3pairs.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")
        
        # Save summary statistics
        summary_path = os.path.join(self.results_dir, "summary_statistics_3pairs.csv")
        summary_metrics = ['EN', 'SD']
        summary_stats = df[summary_metrics].describe()
        summary_stats.to_csv(summary_path)
        print(f"💾 Summary statistics saved to: {summary_path}")
        
        # Save individual pair results as text
        text_path = os.path.join(self.results_dir, "individual_results_3pairs.txt")
        with open(text_path, 'w') as f:
            f.write("Individual Pair Results - First 3 pairs from 21_pairs_tno Dataset\n")
            f.write("="*60 + "\n\n")
            
            for idx, row in df.iterrows():
                pair_id = int(row['pair_id'])
                f.write(f"Pair {pair_id:2d}:\n")
                f.write(f"  EN: {row['EN']:.4f}\n")
                f.write(f"  SD: {row['SD']:.4f}\n\n")
        
        print(f"💾 Individual results saved to: {text_path}")

def main():
    """Main evaluation function"""
    evaluator = BatchFusionEvaluator()
    results_df = evaluator.evaluate_all_pairs()
    
    print(f"\n🎯 Evaluation complete!")
    print(f"📁 All results saved in: {evaluator.results_dir}")
    print(f"📊 {len(results_df)} pairs successfully evaluated")
    
    return results_df

if __name__ == "__main__":
    results = main()