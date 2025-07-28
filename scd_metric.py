#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sum of Correlations of Differences (SCD) Metric Implementation - Multiple Variants

SCD measures the correlation between the differences of the fused image 
with respect to the source images. Higher SCD values indicate better 
fusion performance by preserving the complementary information from both sources.

This file contains multiple SCD calculation variants to match different paper implementations.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import convolve2d
import cv2

def calculate_correlation(x, y):
    """
    Calculate Pearson correlation coefficient between two arrays (Original Method)
    
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

def calculate_correlation_scipy(x, y):
    """
    Calculate correlation using scipy.stats.pearsonr
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Remove NaN and Inf values
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    if np.sum(mask) < 2:  # Need at least 2 points for correlation
        return 0.0
    
    try:
        corr, _ = pearsonr(x_flat[mask], y_flat[mask])
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

def calculate_correlation_normalized(x, y):
    """
    Calculate correlation with enhanced normalization
    """
    x_flat = x.flatten().astype(np.float64)
    y_flat = y.flatten().astype(np.float64)
    
    # Normalize to [0, 1] range
    x_min, x_max = np.min(x_flat), np.max(x_flat)
    y_min, y_max = np.min(y_flat), np.max(y_flat)
    
    if x_max - x_min > 0:
        x_norm = (x_flat - x_min) / (x_max - x_min)
    else:
        x_norm = x_flat
        
    if y_max - y_min > 0:
        y_norm = (y_flat - y_min) / (y_max - y_min)
    else:
        y_norm = y_flat
    
    return calculate_correlation(x_norm.reshape(x.shape), y_norm.reshape(y.shape))

def calculate_correlation_robust(x, y):
    """
    Calculate robust correlation (less sensitive to outliers)
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate median absolute deviation (MAD) for robust scaling
    x_median = np.median(x_flat)
    y_median = np.median(y_flat)
    
    x_mad = np.median(np.abs(x_flat - x_median))
    y_mad = np.median(np.abs(y_flat - y_median))
    
    if x_mad == 0:
        x_scaled = x_flat - x_median
    else:
        x_scaled = (x_flat - x_median) / x_mad
        
    if y_mad == 0:
        y_scaled = y_flat - y_median
    else:
        y_scaled = (y_flat - y_median) / y_mad
    
    return calculate_correlation(x_scaled.reshape(x.shape), y_scaled.reshape(y.shape))

def calculate_scd(ir_img, vis_img, fused_img):
    """
    Calculate Sum of Correlations of Differences (SCD) metric - Original Method
    
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

def calculate_scd_variant1_scipy(ir_img, vis_img, fused_img):
    """
    SCD Variant 1: Using scipy correlation calculation
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate correlations using scipy
    corr1 = calculate_correlation_scipy(diff_fused_ir, diff_vis_ir)
    corr2 = calculate_correlation_scipy(diff_fused_vis, diff_ir_vis)
    
    # SCD is the sum of both correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant2_normalized(ir_img, vis_img, fused_img):
    """
    SCD Variant 2: Using normalized correlation calculation
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate correlations with normalization
    corr1 = calculate_correlation_normalized(diff_fused_ir, diff_vis_ir)
    corr2 = calculate_correlation_normalized(diff_fused_vis, diff_ir_vis)
    
    # SCD is the sum of both correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant3_robust(ir_img, vis_img, fused_img):
    """
    SCD Variant 3: Using robust correlation calculation
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate robust correlations
    corr1 = calculate_correlation_robust(diff_fused_ir, diff_vis_ir)
    corr2 = calculate_correlation_robust(diff_fused_vis, diff_ir_vis)
    
    # SCD is the sum of both correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant4_absolute(ir_img, vis_img, fused_img):
    """
    SCD Variant 4: Using absolute values of correlations
    Some papers use absolute correlation values
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate correlations and take absolute values
    corr1 = abs(calculate_correlation(diff_fused_ir, diff_vis_ir))
    corr2 = abs(calculate_correlation(diff_fused_vis, diff_ir_vis))
    
    # SCD is the sum of both absolute correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant5_squared(ir_img, vis_img, fused_img):
    """
    SCD Variant 5: Using squared correlations
    Some implementations square the correlations
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate correlations and square them
    corr1 = calculate_correlation(diff_fused_ir, diff_vis_ir) ** 2
    corr2 = calculate_correlation(diff_fused_vis, diff_ir_vis) ** 2
    
    # SCD is the sum of both squared correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant6_weighted(ir_img, vis_img, fused_img):
    """
    SCD Variant 6: Weighted correlation based on image variance
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_img - ir_img      # F - A
    diff_vis_ir = vis_img - ir_img          # B - A
    diff_fused_vis = fused_img - vis_img    # F - B
    diff_ir_vis = ir_img - vis_img          # A - B
    
    # Calculate weights based on variance
    var_ir = np.var(ir_img)
    var_vis = np.var(vis_img)
    total_var = var_ir + var_vis
    
    if total_var > 0:
        weight1 = var_vis / total_var  # Weight for IR correlation
        weight2 = var_ir / total_var   # Weight for VIS correlation
    else:
        weight1 = weight2 = 0.5
    
    # Calculate weighted correlations
    corr1 = calculate_correlation(diff_fused_ir, diff_vis_ir)
    corr2 = calculate_correlation(diff_fused_vis, diff_ir_vis)
    
    # Weighted SCD
    scd = weight1 * corr1 + weight2 * corr2
    
    return scd

def calculate_scd_variant7_enhanced(ir_img, vis_img, fused_img):
    """
    SCD Variant 7: Enhanced method with histogram equalization preprocessing
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to uint8 for histogram equalization, then back to float
    ir_eq = cv2.equalizeHist(ir_img.astype(np.uint8)).astype(np.float64)
    vis_eq = cv2.equalizeHist(vis_img.astype(np.uint8)).astype(np.float64)
    fused_eq = cv2.equalizeHist(fused_img.astype(np.uint8)).astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = fused_eq - ir_eq      # F - A
    diff_vis_ir = vis_eq - ir_eq          # B - A
    diff_fused_vis = fused_eq - vis_eq    # F - B
    diff_ir_vis = ir_eq - vis_eq          # A - B
    
    # Calculate correlations
    corr1 = calculate_correlation(diff_fused_ir, diff_vis_ir)
    corr2 = calculate_correlation(diff_fused_vis, diff_ir_vis)
    
    # SCD is the sum of both correlations
    scd = corr1 + corr2
    
    return scd

def calculate_scd_variant8_covariance_based(ir_img, vis_img, fused_img):
    """
    SCD Variant 8: Using covariance matrix approach
    """
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Convert to float for calculations
    ir_img = ir_img.astype(np.float64)
    vis_img = vis_img.astype(np.float64)
    fused_img = fused_img.astype(np.float64)
    
    # Calculate difference images
    diff_fused_ir = (fused_img - ir_img).flatten()      # F - A
    diff_vis_ir = (vis_img - ir_img).flatten()          # B - A
    diff_fused_vis = (fused_img - vis_img).flatten()    # F - B
    diff_ir_vis = (ir_img - vis_img).flatten()          # A - B
    
    # Calculate covariance matrices
    try:
        cov_matrix1 = np.cov(diff_fused_ir, diff_vis_ir)
        cov_matrix2 = np.cov(diff_fused_vis, diff_ir_vis)
        
        # Extract correlations from covariance matrices
        std1_1 = np.sqrt(cov_matrix1[0,0])
        std1_2 = np.sqrt(cov_matrix1[1,1])
        std2_1 = np.sqrt(cov_matrix2[0,0])
        std2_2 = np.sqrt(cov_matrix2[1,1])
        
        if std1_1 > 0 and std1_2 > 0:
            corr1 = cov_matrix1[0,1] / (std1_1 * std1_2)
        else:
            corr1 = 0.0
            
        if std2_1 > 0 and std2_2 > 0:
            corr2 = cov_matrix2[0,1] / (std2_1 * std2_2)
        else:
            corr2 = 0.0
    except:
        corr1 = corr2 = 0.0
    
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
    scd = calculate_scd(ir_img, vis_img, fused_img)
    return scd + 2  # Shift range from [-2, 2] to [0, 4], then normalize if needed 
