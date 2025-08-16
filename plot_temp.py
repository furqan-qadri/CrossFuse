import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('./models/loss/temperature_log.txt')
epochs = data['Epoch']
temps = data['Temperature']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, temps, 'b-o', linewidth=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='Original (1.0)')
plt.xlabel('Epoch')
plt.ylabel('Temperature')
plt.title('Temperature Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('temperature_evolution.png', dpi=300)
plt.show()

print(f"Final temperature: {temps.iloc[-1]:.4f}")
