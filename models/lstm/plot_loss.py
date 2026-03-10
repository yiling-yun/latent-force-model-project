import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Replace this with the actual name of the CSV file in your results folder
csv_path = "results/lstm_force_batch1_lr0.003_hd64_nl2_loss.csv"
# Load the data
df = pd.read_csv(csv_path)

# Find the exact epoch where Validation Loss was lowest
best_epoch = df.loc[df['ValLoss'].idxmin()]['Epoch']
lowest_val_loss = df['ValLoss'].min()

print(f"The model reached its absolute best state at Epoch {int(best_epoch)}")
print(f"Lowest Validation Loss: {lowest_val_loss:.4f}")

# Plotting
plt.figure(figsize=(10, 6))

# Plot Train and Val Loss
plt.plot(df['Epoch'], df['TrainLoss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(df['Epoch'], df['ValLoss'], label='Validation Loss', color='red', linewidth=2)

# Mark the sweet spot
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({int(best_epoch)})')
plt.plot(best_epoch, lowest_val_loss, marker='o', color='green', markersize=8)

plt.title('Training vs Validation Loss (Finding the Overfit Point)')
plt.xlabel('Epoch')
plt.ylabel('Loss (Lower is Better)')
plt.legend()
plt.grid(True)
plt.show()