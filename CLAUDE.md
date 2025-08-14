# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (for transformer models)
- opencv-python >= 4.5.0
- datasets >= 2.0.0 (for KAIST dataset loading)

### GPU Configuration
The codebase is designed for CUDA GPU training. Training scripts set `CUDA_VISIBLE_DEVICES="0"` by default.

## Core Architecture

### Two-Stage Training Process
1. **Auto-encoder pre-training**: Train separate autoencoders for IR and visible images
2. **Cross-attention fusion training**: Train the CrossFuse transformer with pre-trained encoders

### Key Network Components
- **Auto_Encoder_single** (`network/net_autoencoder.py`): Single-modal autoencoder for feature extraction
- **Trans_FuseNet** (`network/net_conv_trans.py`): Main fusion network with cross-attention mechanism
- **cross_encoder** (`network/transformer_cam.py`): Cross-attention transformer module implementing the novel CAM mechanism

### Configuration Files
- `args_auto.py`: Autoencoder training parameters
- `args_trans.py`: CrossFuse fusion network parameters and paths

## Training Commands

### Stage 1: Train Autoencoders
```bash
python train_autoencoder.py
```
Trains separate autoencoders for IR and visible modalities using the KAIST dataset.

### Stage 2: Train CrossFuse Network
```bash
python train_conv_trans.py
```
Trains the cross-attention fusion network using pre-trained autoencoders.

## Testing and Inference

### Single Image Fusion
```bash
python test_single.py
```
Fuse individual IR/visible image pairs.

### Batch Testing
```bash
python test_conv_trans.py
```
Process multiple images using trained models.

### Evaluation Scripts
```bash
# TNO dataset evaluation (21 pairs)
python evaluate_21pairs_tno.py

# SCD metric variants evaluation  
python evaluate_scd_variants.py

# Latest model evaluation
python eval_latests.py
```

## Dataset Structure

### KAIST Dataset (Training)
```
kaist_dataset/
├── kaist_train/
│   └── set00/
│       └── V000/
│           └── lwir/          # IR images
```

### Test Images
```
images/
├── 21_pairs_tno/
│   ├── ir/                    # IR test images
│   └── vis/                   # Visible test images
```

### Model Storage
```
models/
├── autoencoder/               # Pre-trained autoencoders
└── transfuse/                 # CrossFuse fusion models
```

## Key Model Files
- `auto_encoder_epoch_4_ir.model`: Pre-trained IR autoencoder
- `auto_encoder_epoch_4_vi.model`: Pre-trained visible autoencoder  
- `fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model`: Trained CrossFuse model

## Evaluation Metrics
The framework implements comprehensive fusion quality metrics:
- **Entropy (EN)**: Information content measurement
- **Standard Deviation (SD)**: Contrast evaluation on histogram-equalized images
- **Spatial frequency measures**: Gradients and local variations
- **Information-theoretic metrics**: FMI (Feature Mutual Information), MI (Mutual Information)
- **SCD (Sum of Correlations of Differences)**: Multi-scale analysis metric

## Network Configuration Parameters

### Auto-encoder Config
```python
custom_config_auto = {
    "in_channels": 1,
    "out_channels": 1,
    "en_out_channels1": 32,
    "en_out_channels": 64,
    "num_layers": 3,
    "dense_out": 128,
    "part_out": 128,
    "train_flag": True,
}
```

### CrossFuse Config  
```python
custom_config = {
    "img_size": 32,
    "patch_size": 2,
    "depth_self": 1,      # Self-attention depth
    "depth_cross": 1,     # Cross-attention depth
}
```

## Output Structure
Results are saved to `output/crossfuse_test/` with organized subdirectories for different test datasets.