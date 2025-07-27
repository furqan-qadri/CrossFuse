#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Fusion Evaluation Script - 6 Key Metrics
Based on standard references for image fusion evaluation
"""

import numpy as np
import cv2
from scipy.fft import dct
from scipy.stats import entropy
import os

class FusionEvaluator:
    """
    Implements 6 key image fusion evaluation metrics based on standard references
    """
    
    def __init__(self):
        pass
    
    def load_image(self, path):
        """Load image as grayscale numpy array normalized to [0, 255]"""
        if isinstance(path, str):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return img.astype(np.float64)
        else:
            # Handle tensor input
            try:
                import torch
                if torch.is_tensor(path):
                    img = path.detach().cpu().numpy()
                    if img.ndim == 4:  # (B, C, H, W)
                        img = img[0, 0]
                    elif img.ndim == 3:  # (C, H, W)
                        img = img[0]
                else:
                    img = np.array(path)
            except ImportError:
                img = np.array(path)
            
            # Normalize to [0, 255]
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.float64)
            return img
    
    def dct2(self, block):
        """2D DCT implementation using 1D DCT"""
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def calculate_entropy(self, img):
        """
        Metric 1: Entropy (EN) ↑
        Reference: [65] Roberts et al. (2008) - Standard information entropy calculation
        """
        img_uint = img.astype(np.uint8)
        hist, _ = np.histogram(img_uint, bins=256, range=[0, 256])
        hist = hist[hist > 0]  # Remove zero entries
        prob = hist / hist.sum()
        return entropy(prob, base=2)
    
    def calculate_standard_deviation(self, img):
        """
        Metric 2: Standard Deviation (SD) ↑
        Reference: [66] Rao (1997) - Statistical standard deviation of pixel intensities
        """
        return np.std(img)
    
    def calculate_mutual_information(self, img1, img2):
        """
        Metric 3: Mutual Information (MI) ↑
        Reference: [67] Qu et al. (2002) - Information-theoretic mutual information
        """
        # Convert to uint8 for histogram calculation
        img1_uint = img1.astype(np.uint8)
        img2_uint = img2.astype(np.uint8)
        
        # Calculate joint histogram
        hist_2d, _, _ = np.histogram2d(img1_uint.ravel(), img2_uint.ravel(), bins=256)
        
        # Calculate marginal histograms
        hist1, _ = np.histogram(img1_uint.ravel(), bins=256)
        hist2, _ = np.histogram(img2_uint.ravel(), bins=256)
        
        # Convert to probabilities
        pxy = hist_2d / float(np.sum(hist_2d))
        px = hist1 / float(np.sum(hist1))
        py = hist2 / float(np.sum(hist2))
        
        # Calculate marginal product
        px_py = px[:, None] * py[None, :]
        
        # Calculate mutual information
        nzs = pxy > 0  # Non-zero entries only
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px_py[nzs] + 1e-10) + 1e-10))
        
        return mi
    
    def calculate_fmi_dct(self, img1, img2, fused):
        """
        Metric 4: Feature-based Mutual Information with DCT (FMI_dct) ↑
        Reference: [68] Haghighat et al. (2011) - DCT-based feature extraction
        """
        def extract_dct_features(img, block_size=8):
            """Extract DCT coefficients as features"""
            h, w = img.shape
            features = []
            
            # Process image in 8x8 blocks (standard DCT block size)
            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    dct_block = self.dct2(block)
                    # Take significant DCT coefficients (upper-left 4x4)
                    features.extend(dct_block[:4, :4].flatten())
            
            return np.array(features)
        
        # Extract DCT features
        feat1 = extract_dct_features(img1)
        feat2 = extract_dct_features(img2)
        feat_fused = extract_dct_features(fused)
        
        # Calculate MI between DCT features
        # Quantize features for histogram calculation
        feat1_q = np.digitize(feat1, bins=np.linspace(feat1.min(), feat1.max(), 64))
        feat2_q = np.digitize(feat2, bins=np.linspace(feat2.min(), feat2.max(), 64))
        feat_fused_q = np.digitize(feat_fused, bins=np.linspace(feat_fused.min(), feat_fused.max(), 64))
        
        # Calculate MI between fused features and source features
        mi1 = self._calculate_discrete_mi(feat_fused_q, feat1_q)
        mi2 = self._calculate_discrete_mi(feat_fused_q, feat2_q)
        
        # Average MI
        fmi_dct = (mi1 + mi2) / 2.0
        return fmi_dct
    
    def calculate_fmi_pixel(self, img1, img2, fused):
        """
        Metric 5: Feature-based Mutual Information with pixel features (FMI_pixel) ↑
        Reference: [68] Haghighat et al. (2011) - Pixel-level feature extraction
        """
        def extract_pixel_features(img, patch_size=3):
            """Extract pixel-level features using local patches"""
            h, w = img.shape
            features = []
            
            pad = patch_size // 2
            img_padded = np.pad(img, pad, mode='reflect')
            
            for i in range(h):
                for j in range(w):
                    patch = img_padded[i:i+patch_size, j:j+patch_size]
                    # Use patch statistics as features
                    features.append([
                        np.mean(patch),
                        np.std(patch),
                        np.max(patch) - np.min(patch)  # Local contrast
                    ])
            
            return np.array(features).flatten()
        
        # Extract pixel-level features
        feat1 = extract_pixel_features(img1)
        feat2 = extract_pixel_features(img2)
        feat_fused = extract_pixel_features(fused)
        
        # Quantize features for MI calculation
        feat1_q = np.digitize(feat1, bins=np.linspace(feat1.min(), feat1.max(), 64))
        feat2_q = np.digitize(feat2, bins=np.linspace(feat2.min(), feat2.max(), 64))
        feat_fused_q = np.digitize(feat_fused, bins=np.linspace(feat_fused.min(), feat_fused.max(), 64))
        
        # Calculate MI between fused features and source features
        mi1 = self._calculate_discrete_mi(feat_fused_q, feat1_q)
        mi2 = self._calculate_discrete_mi(feat_fused_q, feat2_q)
        
        # Average MI
        fmi_pixel = (mi1 + mi2) / 2.0
        return fmi_pixel
    
    def calculate_scd(self, img1, img2, fused):
        """
        Metric 6: Sum of Correlations of Differences (SCD) ↑
        Reference: [69] Aslantas & Bendes (2015) - Correlation analysis of difference images
        """
        # Calculate difference images
        diff1 = fused - img1
        diff2 = fused - img2
        
        # Calculate correlation coefficient between difference images
        diff1_flat = diff1.flatten()
        diff2_flat = diff2.flatten()
        
        # Remove mean for correlation calculation
        diff1_centered = diff1_flat - np.mean(diff1_flat)
        diff2_centered = diff2_flat - np.mean(diff2_flat)
        
        # Calculate correlation
        numerator = np.sum(diff1_centered * diff2_centered)
        denominator = np.sqrt(np.sum(diff1_centered**2) * np.sum(diff2_centered**2))
        
        if denominator == 0:
            correlation = 0
        else:
            correlation = numerator / denominator
        
        # SCD is the sum of correlations (in this case, just the correlation since we have one pair)
        # For multiple image pairs, this would be summed
        scd = abs(correlation)  # Take absolute value as per the reference
        
        return scd
    
    def _calculate_discrete_mi(self, x, y):
        """Helper function to calculate MI for discrete variables"""
        # Ensure we have valid data
        if len(x) == 0 or len(y) == 0:
            return 0
        
        # Create joint histogram
        try:
            hist_2d, _, _ = np.histogram2d(x, y, bins=64)
        except:
            return 0
        
        # Marginal histograms
        hist_x, _ = np.histogram(x, bins=64)
        hist_y, _ = np.histogram(y, bins=64)
        
        # Convert to probabilities
        pxy = hist_2d / float(np.sum(hist_2d))
        px = hist_x / float(np.sum(hist_x))
        py = hist_y / float(np.sum(hist_y))
        
        # Calculate marginal product
        px_py = px[:, None] * py[None, :]
        
        # Calculate MI
        nzs = pxy > 0
        if np.sum(nzs) == 0:
            return 0
        
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px_py[nzs] + 1e-10) + 1e-10))
        return mi
    
    def evaluate_fusion(self, ir_path, vi_path, fused_path):
        """
        Evaluate fusion quality using all 6 metrics
        
        Args:
            ir_path: Path to IR image or tensor
            vi_path: Path to visible image or tensor
            fused_path: Path to fused image or tensor
            
        Returns:
            Dictionary with all metric values
        """
        print("🔍 Loading images...")
        ir_img = self.load_image(ir_path)
        vi_img = self.load_image(vi_path)
        fused_img = self.load_image(fused_path)
        
        print("📊 Calculating fusion evaluation metrics...")
        
        results = {}
        
        # Single image metrics for fused image
        print("   1/6 Calculating Entropy (EN)...")
        results['EN'] = self.calculate_entropy(fused_img)
        
        print("   2/6 Calculating Standard Deviation (SD)...")
        results['SD'] = self.calculate_standard_deviation(fused_img)
        
        # Mutual information metrics
        print("   3/6 Calculating Mutual Information (MI)...")
        mi1 = self.calculate_mutual_information(fused_img, ir_img)
        mi2 = self.calculate_mutual_information(fused_img, vi_img)
        results['MI'] = (mi1 + mi2) / 2.0  # Average MI with both source images
        
        print("   4/6 Calculating Feature MI - DCT (FMI_dct)...")
        results['FMI_dct'] = self.calculate_fmi_dct(ir_img, vi_img, fused_img)
        
        print("   5/6 Calculating Feature MI - Pixel (FMI_pixel)...")
        results['FMI_pixel'] = self.calculate_fmi_pixel(ir_img, vi_img, fused_img)
        
        print("   6/6 Calculating Sum of Correlations of Differences (SCD)...")
        results['SCD'] = self.calculate_scd(ir_img, vi_img, fused_img)
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📊 FUSION EVALUATION RESULTS")
        print("="*60)
        print("All metrics: Higher values indicate better performance ↑")
        print("-"*60)
        
        print(f"1. Entropy (EN):                    {results['EN']:.4f}")
        print(f"2. Standard Deviation (SD):         {results['SD']:.4f}")
        print(f"3. Mutual Information (MI):         {results['MI']:.4f}")
        print(f"4. Feature MI - DCT (FMI_dct):      {results['FMI_dct']:.4f}")
        print(f"5. Feature MI - Pixel (FMI_pixel):  {results['FMI_pixel']:.4f}")
        print(f"6. Sum of Corr. of Diff. (SCD):     {results['SCD']:.4f}")
        
        print("="*60)
        print("📚 References:")
        print("   EN: Roberts et al. (2008)")
        print("   SD: Rao (1997)")
        print("   MI: Qu et al. (2002)")
        print("   FMI: Haghighat et al. (2011)")
        print("   SCD: Aslantas & Bendes (2015)")
        print("="*60)

def evaluate_test_output(test_output_dir="./test_output"):
    """
    Quick function to evaluate your test output
    Assumes you have fused_result.png in test_output directory
    """
    # Default paths
    ir_path = "./kaist_dataset/kaist_train/set00/V000/lwir/I00001.jpg"
    vi_path = "./kaist_dataset/kaist_train/set00/V000/visible/I00001.jpg"
    fused_path = os.path.join(test_output_dir, "fused_result.png")
    
    # Check if files exist
    if not os.path.exists(fused_path):
        print(f"❌ Fused image not found: {fused_path}")
        print("💡 Run your fusion test first to generate the fused image!")
        return None
    
    if not os.path.exists(ir_path):
        print(f"❌ IR image not found: {ir_path}")
        return None
    
    if not os.path.exists(vi_path):
        print(f"❌ Visible image not found: {vi_path}")
        return None
    
    # Evaluate
    evaluator = FusionEvaluator()
    results = evaluator.evaluate_fusion(ir_path, vi_path, fused_path)
    
    # Save results to file
    results_file = os.path.join(test_output_dir, "evaluation_metrics.txt")
    with open(results_file, 'w') as f:
        f.write("Fusion Evaluation Results\n")
        f.write("="*40 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting fusion evaluation...")
    
    # Evaluate your test output
    results = evaluate_test_output()
    
    if results:
        print("\n✅ Evaluation complete!")
        print("💡 Higher values indicate better fusion quality for all metrics.")
    else:
        print("\n❌ Evaluation failed. Check your file paths.")