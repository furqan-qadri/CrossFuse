import os
import cv2
import numpy as np
# Remove skimage.metrics import as we'll implement raw MI
from scipy.stats import entropy

def compute_mutual_information(image1, image2, bins=64):
    """Compute raw Mutual Information between two images using histogram-based entropy.
    Formula: MI = H(image1) + H(image2) - H(image1, image2)
    where H is the Shannon entropy. This avoids normalization to match paper's scale.
    Using 64 bins for histogram to potentially match paper's implementation.
    """
    # Compute histograms without density normalization to get raw counts
    hist_1, _ = np.histogram(image1.flatten(), bins=bins, range=(0, 255))
    hist_2, _ = np.histogram(image2.flatten(), bins=bins, range=(0, 255))
    hist_joint, _, _ = np.histogram2d(image1.flatten(), image2.flatten(), bins=bins, range=[[0, 255], [0, 255]])
    
    # Normalize to probabilities
    total_pixels = image1.size
    hist_1 = hist_1 / total_pixels
    hist_2 = hist_2 / total_pixels
    hist_joint = hist_joint / total_pixels
    
    # Compute entropies (handle zero probabilities by adding small epsilon)
    h1 = entropy(hist_1 + 1e-10, base=2)
    h2 = entropy(hist_2 + 1e-10, base=2)
    h_joint = entropy(hist_joint.flatten() + 1e-10, base=2)
    
    # Compute raw mutual information
    mi = h1 + h2 - h_joint
    return mi

def compute_scd(ir_image, vis_image, fused_image):
    """Compute Sum of the Correlations of Differences (SCD) using a simple approach.
    Formula: SCD = Corr(F-IR, VIS-IR) + Corr(F-VIS, IR-VIS)
    Purpose: Measures how well the fused image preserves information from source images.
    Using absolute correlations to match typical implementations and get closer to paper value.
    Handling potential NaN values in correlations by checking and setting to 0.
    """
    diff_f_ir = fused_image.astype(float) - ir_image.astype(float)
    diff_vis_ir = vis_image.astype(float) - ir_image.astype(float)
    diff_f_vis = fused_image.astype(float) - vis_image.astype(float)
    diff_ir_vis = ir_image.astype(float) - vis_image.astype(float)
    
    corr1_val = np.corrcoef(diff_f_ir.flatten(), diff_vis_ir.flatten())[0, 1]
    corr2_val = np.corrcoef(diff_f_vis.flatten(), diff_ir_vis.flatten())[0, 1]
    corr1 = abs(corr1_val) if not np.isnan(corr1_val) else 0.0
    corr2 = abs(corr2_val) if not np.isnan(corr2_val) else 0.0
    
    scd = corr1 + corr2
    return scd

def load_image(path):
    """Load an image in grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def evaluate_tno_pairs(ir_dir, vis_dir, fused_dir):
    """Evaluate MI and SCD for TNO image pairs."""
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
    results = []
    
    print(f"Evaluating {len(ir_files)} image pairs...")
    for ir_file in ir_files:
        ir_path = os.path.join(ir_dir, ir_file)
        vis_file = ir_file.replace('IR', 'VIS')  # Assuming naming convention
        vis_path = os.path.join(vis_dir, vis_file)
        fused_path = os.path.join(fused_dir, 'results_transfuse_' + ir_file)
        
        if not os.path.exists(vis_path) or not os.path.exists(fused_path):
            print(f"Skipping {ir_file}: Missing VIS or fused image.")
            continue
        
        ir_img = load_image(ir_path)
        vis_img = load_image(vis_path)
        fused_img = load_image(fused_path)
        
        if ir_img is None or vis_img is None or fused_img is None:
            print(f"Skipping {ir_file}: Failed to load one or more images.")
            continue
        
        # Compute MI between fused and source images
        mi_ir_fused = compute_mutual_information(ir_img, fused_img)
        mi_vis_fused = compute_mutual_information(vis_img, fused_img)
        total_mi = mi_ir_fused + mi_vis_fused
        
        # Compute SCD
        scd = compute_scd(ir_img, vis_img, fused_img)
        
        results.append({
            'image': ir_file,
            'MI_IR_Fused': mi_ir_fused,
            'MI_VIS_Fused': mi_vis_fused,
            'Total_MI': total_mi,
            'SCD': scd
        })
        print(f"Processed {ir_file}: Total MI = {total_mi:.4f}, SCD = {scd:.4f}")
    
    return results

def print_summary(results):
    """Print summary of evaluation results."""
    if not results:
        print("No results to summarize.")
        return
    
    mi_values = [r['Total_MI'] for r in results]
    scd_values = [r['SCD'] for r in results]
    
    avg_mi = np.mean(mi_values)
    avg_scd = np.mean(scd_values)
    
    print("\nEvaluation Summary for TNO Image Pairs:")
    print(f"Average Total Mutual Information (MI): {avg_mi:.4f}")
    print(f"Average Sum of Correlations of Differences (SCD): {avg_scd:.4f}")
    print(f"Number of evaluated pairs: {len(results)}")

if __name__ == '__main__':
    ir_dir = './images/21_pairs_tno/ir'
    vis_dir = './images/21_pairs_tno/vis'
    fused_dir = './output/crossfuse_test/21_pairs_tno_transfuse'
    
    if not os.path.exists(ir_dir) or not os.path.exists(vis_dir) or not os.path.exists(fused_dir):
        print("One or more directories not found. Please check paths.")
    else:
        results = evaluate_tno_pairs(ir_dir, vis_dir, fused_dir)
        print_summary(results) 