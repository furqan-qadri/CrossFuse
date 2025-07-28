#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sum of Correlations of Differences (SCD) Metric Implementation

SCD measures the correlation between the differences of the fused image 
with respect to the source images. Higher SCD values indicate better 
fusion performance by preserving the complementary information from both sources.
"""

import numpy as np

def calculate_correlation(x, y):
    """
    Calculate Pearson correlation coefficient between two arrays
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        float: Correlation coefficient
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate means
    mean_x = np.mean(x_flat)
    mean_y = np.mean(y_flat)
    
    # Calculate correlation coefficient
    numerator = np.sum((x_flat - mean_x) * (y_flat - mean_y))
    denominator = np.sqrt(np.sum((x_flat - mean_x)**2) * np.sum((y_flat - mean_y)**2))
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return correlation

def calculate_scd(ir_img, vis_img, fused_img):
    """
    Calculate Sum of Correlations of Differences (SCD) metric
    
    Formula: SCD = Corr(F-A, B-A) + Corr(F-B, A-B)
    where F=fused image, A=IR image, B=visible image
    
    Args:
        ir_img: IR (infrared) source image as numpy array
        vis_img: Visible source image as numpy array
        fused_img: Fused result image as numpy array
        
    Returns:
        float: SCD score (higher is better)
    """
    
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to grayscale if needed
    if len(ir_img.shape) == 3:
        ir_img = np.mean(ir_img, axis=2)
    if len(vis_img.shape) == 3:
        vis_img = np.mean(vis_img, axis=2)
    if len(fused_img.shape) == 3:
        fused_img = np.mean(fused_img, axis=2)
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate correlations between difference images
    corr1 = calculate_correlation(diff_fused_ir, diff_vis_ir)   # Corr(F-A, B-A)
    corr2 = calculate_correlation(diff_fused_vis, diff_ir_vis)  # Corr(F-B, A-B)
    
    # SCD is the sum of both correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_normalized(ir_img, vis_img, fused_img):
    """
    Calculate normalized SCD metric (range [0, 2])
    
    This version ensures the SCD value is always positive by adding 1 to each correlation
    before summing, resulting in a range of [0, 2] instead of [-2, 2].
    
    Args:
        ir_img: IR (infrared) source image as numpy array
        vis_img: Visible source image as numpy array
        fused_img: Fused result image as numpy array
        
    Returns:
        float: Normalized SCD score (higher is better, range [0, 2])
    """
    
    # Calculate standard SCD
    # scd = calculate_scd(ir_img, vis_img, fused_img)
    
    # Normalize to [0, 2] range
    # scd_normalized=scd
    
    # return scd 
