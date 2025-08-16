# Multi-Head Temperature Specialization Tracking

This implementation provides simple and clean temperature tracking for the multi-head attention specialization during training.

## Files Created

1. **`temperature_tracker.py`** - Main tracking system
2. **`plot_temperature_graphs.py`** - Graph generation script
3. **Modified `train_conv_trans.py`** - Integrated tracking into training loop

## How It Works

### During Training

The system automatically:

- Extracts temperature values from all cross-attention modules at the end of each epoch
- Saves data to timestamped text files in `temperature_logs/` directory
- Prints specialization status every 5 epochs
- Creates a final training summary

### Generated Files

```
temperature_logs/
├── head_temperatures_YYYYMMDD_HHMMSS.txt    # Individual head temperatures per epoch
├── temperature_variance_YYYYMMDD_HHMMSS.txt  # Variance timeline data
└── training_summary.txt                       # Final specialization summary
```

### Graph Generation

Run `python plot_temperature_graphs.py` after training to create:

1. **Head Temperature Bar Chart** - Shows final temperature values for all 16 heads

   - Red bars: Sharp focus (τ < 0.7)
   - Blue bars: Global context (τ > 1.3)
   - Green bars: Balanced (0.7 ≤ τ ≤ 1.3)

2. **Specialization Timeline** - Shows temperature variance evolution over epochs
   - Demonstrates when specialization emerges during training
   - Shows progression from uniform (low variance) to specialized (high variance)

## Usage

### Training with Temperature Tracking

Just run your normal training command:

```bash
python train_conv_trans.py
```

The temperature tracking is now automatically integrated and will:

- Create `temperature_logs/` directory
- Save data files during training
- Print specialization updates every 5 epochs

### Creating Graphs

After training completes:

```bash
python plot_temperature_graphs.py
```

This will:

- Load the most recent temperature log files
- Generate both required graphs
- Save them as PNG files in `temperature_graphs/` directory
- Display the graphs on screen

## Data Format

### Head Temperatures File

```
# Format: epoch,head_1,head_2,...,head_16
1,1.0000,1.0000,1.0000,1.0000,...
2,0.9845,1.0234,0.9876,1.0123,...
```

### Variance Timeline File

```
# Format: epoch,variance,mean_temp,min_temp,max_temp
1,0.000000,1.0000,1.0000,1.0000
2,0.001234,1.0023,0.9845,1.0234
```

## Key Features

- **Minimal Code**: Only essential tracking functionality
- **Real Training Data**: Uses actual temperature values from your model
- **Automatic Integration**: No manual intervention needed during training
- **Clean Output**: Well-formatted text files for easy analysis
- **Ready-to-Use Graphs**: Publication-ready visualization

## Expected Results

The graphs will demonstrate:

- **Specialization Evidence**: Different heads converging to different temperature ranges
- **Timeline Discovery**: Clear progression showing when specialization emerges
- **Categorical Distribution**: Visual proof of sharp/balanced/broad head categories

This provides clear visual evidence that the multi-head temperature mechanism enables automatic specialization of attention heads during training.
