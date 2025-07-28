#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Mutual Information (FMI) Metric Implementation

Based on: M.B.A. Haghighat et al., "A non-reference image fusion metric based on 
mutual information of image features", Computer & Electrical Engineering, 2011

FMI measures the mutual information between edge features of the fused image 
and the source images. Higher FMI values indicate better fusion performance.
"""

import numpy as np
from scipy.ndimage import sobel

def extract_edge_features(image):
    """
    Extract edge features using Sobel operators
    
    Args:
        image: Input image as numpy array
        
    Returns:
        numpy array: Edge magnitude features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # Apply Sobel operators
    sobel_x = sobel(image, axis=1)
    sobel_y = sobel(image, axis=0) 
    
    # Calculate edge magnitude
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return edge_magnitude

def calculate_mutual_information(x, y, bins=256):
    """
    Calculate mutual information between two feature arrays
    
    Args:
        x: First feature array
        y: Second feature array  
        bins: Number of histogram bins
        
    Returns:
        float: Mutual information value
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Normalize to [0, 255] range for histogram
    x_norm = ((x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-8) * 255).astype(np.uint8)
    y_norm = ((y_flat - y_flat.min()) / (y_flat.max() - y_flat.min() + 1e-8) * 255).astype(np.uint8)
    
    # Create 2D histogram
    hist_2d, _, _ = np.histogram2d(x_norm, y_norm, bins=bins, range=[[0, 255], [0, 255]])
    
    # Add small epsilon to avoid log(0)
    hist_2d = hist_2d + 1e-10
    
    # Normalize to get joint probability
    pxy = hist_2d / hist_2d.sum()
    
    # Calculate marginal probabilities
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 1e-10:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    
    return mi

def calculate_fmi(ir_img, vis_img, fused_img):
    """
    Calculate Feature Mutual Information (FMI) metric
    
    Formula: FMI = MI(F_features, A_features) + MI(F_features, B_features)
    where F=fused image, A=IR image, B=visible image
    
    Args:
        ir_img: IR (infrared) source image as numpy array
        vis_img: Visible source image as numpy array
        fused_img: Fused result image as numpy array
        
    Returns:
        float: FMI score (higher is better)
    """
    
    # Ensure images are the same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Extract edge features from each image
    ir_features = extract_edge_features(ir_img)
    vis_features = extract_edge_features(vis_img) 
    fused_features = extract_edge_features(fused_img)
    
    # Calculate mutual information between fused features and each source's features
    mi_fused_ir = calculate_mutual_information(fused_features, ir_features)
    mi_fused_vis = calculate_mutual_information(fused_features, vis_features)
    
    # FMI is the sum of mutual information with both source image features
    fmi = mi_fused_ir + mi_fused_vis
    
    return fmi 