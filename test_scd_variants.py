#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for SCD Variants Comparison
This script tests all SCD calculation methods to find which one matches the paper's value of 1.7659
"""

import numpy as np
import cv2
import os
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

def load_test_images():
    """
    Load test images from the TNO dataset or output directory
    """
    # Try to load from the same test images used in evaluation
    ir_path = "./images/21_pairs_tno/ir/IR1.png"
    vis_path = "./images/21_pairs_tno/vis/VIS1.png"
    fused_path = "./output/crossfuse_test/21_pairs_tno_transfuse/results_transfuse_IR1.png"
    
    # Alternative paths if the above don't exist
    alt_paths = [
        ("./test_output/crossfuse_test/21_pairs_tno_transfuse_cpu/results_transfuse_IR1.png", "fused"),
        ("./images/21_pairs_tno/ir/IR1.bmp", "ir"),
        ("./images/21_pairs_tno/vis/VIS1.bmp", "vis"),
    ]
    
    try:
        # Try primary paths first
        if os.path.exists(ir_path) and os.path.exists(vis_path) and os.path.exists(fused_path):
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            fused_img = cv2.imread(fused_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Try alternative paths
            print("Primary paths not found, trying alternatives...")
            # You might need to adjust these paths based on your actual file structure
            return None, None, None
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None, None
    
    if ir_img is not None and vis_img is not None and fused_img is not None:
        print(f"Successfully loaded images:")
        print(f"  IR shape: {ir_img.shape}")
        print(f"  VIS shape: {vis_img.shape}")
        print(f"  Fused shape: {fused_img.shape}")
        return ir_img, vis_img, fused_img
    else:
        print("Failed to load test images")
        return None, None, None

def create_synthetic_test_data():
    """
    Create synthetic test data for testing SCD variants
    """
    print("Creating synthetic test data...")
    
    # Create synthetic images (256x256)
    size = 256
    
    # IR image: gradient with some noise
    ir_img = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        ir_img[i, :] = (i / size) * 255
    ir_img += np.random.normal(0, 10, (size, size))  # Add noise
    ir_img = np.clip(ir_img, 0, 255)
    
    # VIS image: different pattern
    vis_img = np.zeros((size, size), dtype=np.float64)
    for j in range(size):
        vis_img[:, j] = (j / size) * 255
    vis_img += np.random.normal(0, 10, (size, size))  # Add noise
    vis_img = np.clip(vis_img, 0, 255)
    
    # Fused image: combination of both with some enhancement
    fused_img = 0.6 * ir_img + 0.4 * vis_img
    fused_img += np.random.normal(0, 5, (size, size))  # Less noise
    fused_img = np.clip(fused_img, 0, 255)
    
    return ir_img.astype(np.uint8), vis_img.astype(np.uint8), fused_img.astype(np.uint8)

def test_all_scd_variants(ir_img, vis_img, fused_img):
    """
    Test all SCD calculation variants and compare results
    """
    print("\n" + "="*80)
    print("TESTING ALL SCD VARIANTS")
    print("="*80)
    print(f"Target value from paper: 1.7659")
    print("-"*80)
    
    # Define all SCD variants with descriptions
    variants = [
        ("Original Method", calculate_scd),
        ("Variant 1: Scipy Correlation", calculate_scd_variant1_scipy),
        ("Variant 2: Normalized Correlation", calculate_scd_variant2_normalized),
        ("Variant 3: Robust Correlation", calculate_scd_variant3_robust),
        ("Variant 4: Absolute Correlations", calculate_scd_variant4_absolute),
        ("Variant 5: Squared Correlations", calculate_scd_variant5_squared),
        ("Variant 6: Weighted Correlation", calculate_scd_variant6_weighted),
        ("Variant 7: Histogram Equalized", calculate_scd_variant7_enhanced),
        ("Variant 8: Covariance Based", calculate_scd_variant8_covariance_based),
        ("Normalized SCD (+2)", calculate_scd_normalized),
    ]
    
    results = []
    target_value = 1.7659
    
    for name, func in variants:
        try:
            scd_value = func(ir_img, vis_img, fused_img)
            diff_from_target = abs(scd_value - target_value)
            results.append((name, scd_value, diff_from_target))
            print(f"{name:<35}: {scd_value:>8.4f} (diff: {diff_from_target:>6.4f})")
        except Exception as e:
            print(f"{name:<35}: ERROR - {str(e)}")
            results.append((name, float('nan'), float('inf')))
    
    print("-"*80)
    
    # Sort by closest to target
    valid_results = [(name, val, diff) for name, val, diff in results if not np.isnan(val)]
    valid_results.sort(key=lambda x: x[2])  # Sort by difference from target
    
    print("\nRANKING (closest to target first):")
    print("-"*80)
    for i, (name, value, diff) in enumerate(valid_results, 1):
        print(f"{i:>2}. {name:<35}: {value:>8.4f} (diff: {diff:>6.4f})")
    
    return results

def analyze_correlation_components(ir_img, vis_img, fused_img):
    """
    Analyze the individual correlation components of SCD
    """
    print("\n" + "="*80)
    print("ANALYZING SCD CORRELATION COMPONENTS")
    print("="*80)
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    from scd_metric import calculate_correlation
    
    # Calculate individual correlations
    corr1 = calculate_correlation(diff_fused_ir, diff_vis_ir)   # Corr(F-A, B-A)
    corr2 = calculate_correlation(diff_fused_vis, diff_ir_vis)  # Corr(F-B, A-B)
    
    print(f"Correlation 1 [Corr(F-A, B-A)]: {corr1:.6f}")
    print(f"Correlation 2 [Corr(F-B, A-B)]: {corr2:.6f}")
    print(f"SCD = Corr1 + Corr2 = {corr1 + corr2:.6f}")
    
    # Statistics of difference images
    print(f"\nDifference Image Statistics:")
    print(f"  (F-A) - mean: {np.mean(diff_fused_ir):.4f}, std: {np.std(diff_fused_ir):.4f}")
    print(f"  (B-A) - mean: {np.mean(diff_vis_ir):.4f}, std: {np.std(diff_vis_ir):.4f}")
    print(f"  (F-B) - mean: {np.mean(diff_fused_vis):.4f}, std: {np.std(diff_fused_vis):.4f}")
    print(f"  (A-B) - mean: {np.mean(diff_ir_vis):.4f}, std: {np.std(diff_ir_vis):.4f}")
    
    # Image statistics
    print(f"\nOriginal Image Statistics:")
    print(f"  IR  - mean: {np.mean(ir_img):.4f}, std: {np.std(ir_img):.4f}")
    print(f"  VIS - mean: {np.mean(vis_img):.4f}, std: {np.std(vis_img):.4f}")
    print(f"  FUSED - mean: {np.mean(fused_img):.4f}, std: {np.std(fused_img):.4f}")

def main():
    print("SCD Variants Testing Script")
    print("="*80)
    
    # Try to load real test images first
    ir_img, vis_img, fused_img = load_test_images()
    
    if ir_img is None:
        print("Real images not available, using synthetic test data...")
        ir_img, vis_img, fused_img = create_synthetic_test_data()
    
    # Test all variants
    results = test_all_scd_variants(ir_img, vis_img, fused_img)
    
    # Analyze components
    analyze_correlation_components(ir_img, vis_img, fused_img)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. Look for the variant with the smallest difference from 1.7659")
    print("2. If multiple variants are close, check which method is used in the paper")
    print("3. Consider preprocessing steps (histogram equalization, normalization)")
    print("4. The paper might use a different dataset or image preprocessing")
    print("\nTo test with your actual dataset:")
    print("- Update the image paths in load_test_images() function")
    print("- Run this script with your exact test images")

if __name__ == "__main__":
    main() 