#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutual Information (MI) Metric for Image Fusion

Standard implementation: MI_fusion = (MI(fused, IR) + MI(fused, visible)) / 2
Where MI(X,Y) = H(X) + H(Y) - H(X,Y)
"""

import numpy as np

def calculate_entropy(image, bins=256):
    """
    Calculate Shannon entropy of an image
    """
    # Get histogram
    hist, _ = np.histogram(image.flatten(), bins=bins, range=[0, 255])
    
    # Remove zeros and normalize to get probabilities
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    
    # Calculate entropy: H(X) = -sum(P(x) * log2(P(x)))
    entropy = -np.sum(prob * np.log2(prob))
    
    return entropy

def calculate_joint_entropy(image1, image2, bins=256):
    """
    Calculate joint entropy between two images
    """
    # Create 2D histogram for joint distribution
    hist_2d, _, _ = np.histogram2d(
        image1.flatten(), 
        image2.flatten(), 
        bins=bins, 
        range=[[0, 255], [0, 255]]
    )
    
    # Remove zeros and normalize to get joint probabilities
    hist_2d = hist_2d[hist_2d > 0]
    prob_joint = hist_2d / hist_2d.sum()
    
    # Calculate joint entropy: H(X,Y) = -sum(P(x,y) * log2(P(x,y)))
    joint_entropy = -np.sum(prob_joint * np.log2(prob_joint))
    
    return joint_entropy

def calculate_mutual_information(image1, image2, bins=256):
    """
    Calculate mutual information between two images
    MI(X,Y) = H(X) + H(Y) - H(X,Y)
    """
    # Ensure images are same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = np.mean(image1, axis=2)
    if len(image2.shape) == 3:
        image2 = np.mean(image2, axis=2)
    
    # Ensure valid pixel range [0, 255]
    image1 = np.clip(image1, 0, 255).astype(np.uint8)
    image2 = np.clip(image2, 0, 255).astype(np.uint8)
    
    # Calculate entropies
    h1 = calculate_entropy(image1, bins)
    h2 = calculate_entropy(image2, bins)
    h12 = calculate_joint_entropy(image1, image2, bins)
    
    # MI = H(X) + H(Y) - H(X,Y)
    mi = h1 + h2 - h12
    
    return mi

def calculate_mi(ir_img, vis_img, fused_img, bins=256):
    """
    Calculate MI metric for image fusion evaluation
    
    Standard formula: MI_fusion = (MI(fused, IR) + MI(fused, visible)) / 2
    
    Args:
        ir_img: IR source image as numpy array
        vis_img: Visible source image as numpy array  
        fused_img: Fused result image as numpy array
        bins: Number of histogram bins (default: 256)
        
    Returns:
        float: MI fusion score (higher is better)
    """
    # Ensure all images are same size
    if not (ir_img.shape == vis_img.shape == fused_img.shape):
        raise ValueError("All images must have the same dimensions")
    
    # Calculate MI between fused image and each source
    mi_fused_ir = calculate_mutual_information(fused_img, ir_img, bins)
    mi_fused_vis = calculate_mutual_information(fused_img, vis_img, bins)
    
    # Average MI with both sources
    mi_fusion = (mi_fused_ir + mi_fused_vis) / 2.0
    
    return mi_fusion

def test_mi_metric():
    """
    Test function to verify MI metric implementation
    """
    print("Testing MI metric implementation...")
    
    # Create test images
    np.random.seed(42)
    
    # Test case 1: Identical images should have high MI
    img1 = np.random.rand(100, 100) * 255
    img1_copy = img1.copy()
    mi_identical = calculate_mutual_information(img1, img1_copy)
    print(f"MI between identical images: {mi_identical:.4f}")
    
    # Test case 2: Random images should have lower MI
    img2 = np.random.rand(100, 100) * 255
    img3 = np.random.rand(100, 100) * 255
    mi_random = calculate_mutual_information(img2, img3)
    print(f"MI between random images: {mi_random:.4f}")
    
    # Test case 3: Fusion metric calculation
    ir_test = np.random.rand(50, 50) * 255
    vis_test = np.random.rand(50, 50) * 255
    fused_test = (ir_test + vis_test) / 2  # Simple average fusion
    mi_fusion = calculate_mi(ir_test, vis_test, fused_test)
    print(f"MI for fusion test: {mi_fusion:.4f}")
    
    print("MI metric test completed!")

if __name__ == "__main__":
    test_mi_metric() 