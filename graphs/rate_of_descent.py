import matplotlib.pyplot as plt
import numpy as np

# Define epochs
epochs = np.arange(1, 17)  # 1 to 16 epochs

# Step Decay Learning Rate Schedule (50% reduction every 5 epochs)
def step_decay_lr(epoch, base_lr=0.001):
    """Step decay: 50% reduction at epochs 5, 10, 15"""
    if epoch <= 5:
        return base_lr
    elif epoch <= 10:
        return base_lr * 0.5
    elif epoch <= 15:
        return base_lr * 0.25
    else:  # epoch 16
        return base_lr * 0.125

# Cosine Annealing Learning Rate Schedule
def cosine_annealing_lr(epoch, base_lr=0.001, min_lr=1e-5, total_epochs=16):
    """Cosine annealing from base_lr to min_lr"""
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

# Calculate learning rates for both schedules
step_decay_rates = [step_decay_lr(epoch) for epoch in epochs]
cosine_annealing_rates = [cosine_annealing_lr(epoch) for epoch in epochs]

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(epochs, step_decay_rates, 'b-', linewidth=3, marker='o', markersize=6, 
         label='Step Decay (Every 5 Epochs)', color='#1f77b4')
plt.plot(epochs, cosine_annealing_rates, 'g-', linewidth=3, marker='s', markersize=6, 
         label='Cosine Annealing', color='#2ca02c')

# Customize the plot
plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Learning Rate', fontsize=14, fontweight='bold')
plt.title('Learning Rate Schedule Comparison (16 Epochs)\nStep Decay vs Cosine Annealing', 
          fontsize=16, fontweight='bold')

# Set y-axis to log scale for better visualization
plt.yscale('log')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Customize legend
plt.legend(fontsize=12, loc='upper right')

# Set x-axis ticks for all epochs
plt.xticks(epochs)

# Add annotations for step decay drops
plt.annotate('50% Drop', xy=(5.5, 0.0005), xytext=(7, 0.0008),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

plt.annotate('50% Drop', xy=(10.5, 0.00025), xytext=(12, 0.0004),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

plt.annotate('50% Drop', xy=(15.5, 0.000125), xytext=(14, 0.0002),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

# Add vertical lines to highlight step decay points
plt.axvline(x=5, color='red', linestyle=':', alpha=0.7)
plt.axvline(x=10, color='red', linestyle=':', alpha=0.7)
plt.axvline(x=15, color='red', linestyle=':', alpha=0.7)

# Set axis limits
plt.xlim(1, 16)
plt.ylim(1e-6, 0.002)

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the plot
# plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')