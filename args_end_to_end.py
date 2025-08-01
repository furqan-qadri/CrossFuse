# -*- coding: utf-8 -*-
# Arguments for END-TO-END CrossFuse training
# Based on args_trans.py but optimized for end-to-end training
#
# 🔧 CUDA CONFIGURATION:
# - Set 'cuda = True' for GPU training (recommended)
# - Set 'cuda = False' for CPU training (slower but works on any machine)
# - Auto-detects CUDA availability and falls back to CPU if needed

class Args:
    # Data paths - must be a list like in original args
    path_ir = ['/Users/furqanqadri/Coding/CrossFuse/kaist_dataset/kaist_train/set00/V000/lwir/']
    train_num = 20000  # Number of training pairs
    
    # Network architecture
    channel = 1  # 1 for grayscale, 3 for RGB
    Height = 64
    Width = 64
    
    # Training parameters - ADJUSTED FOR END-TO-END
    epochs = 32  # Same as current max
    batch = 8    # Keep same batch size
    lr = 0.001   # Learning rate (might converge better with slightly lower LR for end-to-end)
    step = 1250  # Print frequency
    
    # Hardware - Auto-detect CUDA or manual override
    cuda = True  # Set to False to force CPU training
    # Auto-detect CUDA availability
    import torch
    if cuda and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, switching to CPU")
        cuda = False
    elif cuda:
        print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"   Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU training (CUDA disabled)")
    
    # Model paths - NO PRE-TRAINED ENCODERS!
    resume_model_trans = None  # Start fusion from scratch too
    resume_model_auto_ir = None  # NO pre-trained IR encoder
    resume_model_auto_vi = None  # NO pre-trained VIS encoder
    
    # Save paths - separate directory for end-to-end results
    save_fusion_model = './models/'
    
    # Other settings
    ssim_weight = [1, 10, 100, 1000, 10000]
    feature_loss_weight = 10
    deep_feature_loss_weight = 10
    
    print("🔄 END-TO-END Training Configuration")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Learning rate: {lr}")
    print(f"   Training pairs: {train_num}")
    print("   Starting ALL components from scratch")
    print("   No pre-trained encoder loading")