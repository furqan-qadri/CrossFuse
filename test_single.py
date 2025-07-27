#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-Only test script for CrossFuse - Test a single IR/Visible pair
Works on macOS without CUDA
"""

import os
import torch
import numpy as np
from torch.autograd import Variable
from network.net_autoencoder import Auto_Encoder_single
from network.net_conv_trans import Trans_FuseNet
from tools import utils

def load_models_cpu():
    """Load all models for CPU-only inference"""
    
    # Auto-Encoder configuration
    custom_config_auto = {
        "in_channels": 1,
        "out_channels": 1,
        "en_out_channels1": 32,
        "en_out_channels": 64,
        "num_layers": 3,
        "dense_out": 128,
        "part_out": 128,
        "train_flag": False,
    }
    
    # Fusion transformer configuration
    custom_config_trans = {
        "en_out_channels1": 32,
        "out_channels": 1,
        "part_out": 128,
        "train_flag": False,
        "img_size": 32,
        "patch_size": 2,
        "depth_self": 1,
        "depth_cross": 1,
        "n_heads": 16,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "p": 0.,
        "attn_p": 0.,
    }
    
    print("🔧 Loading models (CPU-only)...")
    
    # Load IR autoencoder
    print("   Loading IR autoencoder...")
    model_auto_ir = Auto_Encoder_single(**custom_config_auto)
    model_auto_ir.load_state_dict(torch.load("./models/autoencoder/auto_encoder_epoch_4_ir.model", map_location='cpu'))
    model_auto_ir.eval()
    
    # Load Visible autoencoder
    print("   Loading Visible autoencoder...")
    model_auto_vi = Auto_Encoder_single(**custom_config_auto)
    model_auto_vi.load_state_dict(torch.load("./models/autoencoder/auto_encoder_epoch_4_vi.model", map_location='cpu'))
    model_auto_vi.eval()
    
    # Load Fusion transformer
    print("   Loading Fusion transformer...")
    model_trans = Trans_FuseNet(**custom_config_trans)
    model_trans.load_state_dict(torch.load("./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model", map_location='cpu'))
    model_trans.eval()
    
    print("✅ All models loaded successfully on CPU!")
    return model_auto_ir, model_auto_vi, model_trans

def test_single_pair_cpu(ir_image_path, vi_image_path, output_path="./test_output"):
    """Test fusion on a single pair using CPU only"""
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load models
    model_auto_ir, model_auto_vi, model_trans = load_models_cpu()
    
    print(f"📷 Loading images:")
    print(f"   IR: {ir_image_path}")
    print(f"   Visible: {vi_image_path}")
    
    # Check if files exist
    if not os.path.exists(ir_image_path):
        print(f"❌ IR image not found: {ir_image_path}")
        return None
    if not os.path.exists(vi_image_path):
        print(f"❌ Visible image not found: {vi_image_path}")
        return None
    
    # Load and preprocess images (CPU only)
    ir_img = utils.get_train_images([ir_image_path], None, None, flag=False)
    vi_img = utils.get_train_images([vi_image_path], None, None, flag=False)
    
    # Convert to tensors (keep on CPU)
    ir_img = Variable(ir_img, requires_grad=False)
    vi_img = Variable(vi_img, requires_grad=False)
    
    print("🧠 Running fusion on CPU...")
    with torch.no_grad():
        # Extract features using autoencoders
        ir_sh, ir_de = model_auto_ir(ir_img)
        vi_sh, vi_de = model_auto_vi(vi_img)
        
        # Perform fusion
        outputs = model_trans(ir_de, ir_sh, vi_de, vi_sh, shift_flag=True)
        fused_img = outputs['out']
    
    # Save result
    output_filename = os.path.join(output_path, "fused_result.png")
    utils.save_image(fused_img, output_filename)
    
    print(f"✅ Fusion complete! Result saved to: {output_filename}")
    return output_filename

if __name__ == "__main__":
    # Test with your KAIST dataset images
    ir_path = "./kaist_dataset/kaist_train/set00/V000/lwir/I00001.jpg"
    vi_path = "./kaist_dataset/kaist_train/set00/V000/visible/I00001.jpg"
    
    print("🚀 Starting CPU-only fusion test...")
    result = test_single_pair_cpu(ir_path, vi_path)
    
    if result:
        print(f"🎯 Success! Check your fused image: {result}")
    else:
        print("❌ Test failed. Check image paths.")