import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

labels_names_fname = 'data/prep_names.csv'
conf_matrix_fname = 'runs/model-classifier-confrix.npy'
chart_fname = 'runs/top_confusions.svg'

sample_size = 10

df_names = pd.read_csv(labels_names_fname, header=None)
conf_matrix = np.load(conf_matrix_fname)
print("Label names shape:", df_names.shape)
print("Confusion matrix shape:", conf_matrix.shape)
print("Confusion matrix dtype:", conf_matrix.dtype)

class_names = df_names[0].values

# Overall accuracy (macro recall)
macro_accuracy = np.trace(conf_matrix) / len(class_names)
print(f"Macro Average Accuracy / Recall: {macro_accuracy:.4f} ({macro_accuracy*100:.2f}%)")

# Top best recognized classes
diag = np.diagonal(conf_matrix)
top10_idx = np.argsort(diag)[::-1][:sample_size]
print("\nTop 10 Best Recognized Classes:")
for rank, idx in enumerate(top10_idx, 1):
    print(f"{rank}. {class_names[idx]} (ID {idx}): {diag[idx]*100:.2f}%")

# Top worst recognized classes
worst10_idx = np.argsort(diag)[:sample_size]
print("\nTop 10 Worst Recognized Classes:")
for rank, idx in enumerate(worst10_idx, 1):
    print(f"{rank}. {class_names[idx]} (ID {idx}): {diag[idx]*100:.2f}%")

# Top most confused pairs (off-diagonal max)
cm_off = conf_matrix.copy()
np.fill_diagonal(cm_off, 0)

flat_indices = np.argsort(cm_off.ravel())[::-1][:sample_size]
print(f"\nTop {sample_size} Most Frequent Misclassifications (True -> Predicted):")
for rank, flat_idx in enumerate(flat_indices, 1):
    i, j = np.unravel_index(flat_idx, conf_matrix.shape)
    print(f"{rank}. True: '{class_names[i]}' (ID {i}) -> Predicted as: '{class_names[j]}' (ID {j}) | Rate: {conf_matrix[i, j]*100:.2f}%")

# Create a figure for top confusions visualization / summary plot
fig, ax = plt.subplots(figsize=(10, 6))

top_confusions = []
labels = []
for flat_idx in flat_indices[:sample_size]:
    i, j = np.unravel_index(flat_idx, conf_matrix.shape)
    labels.append(f"{class_names[i]} → {class_names[j]}")
    top_confusions.append(conf_matrix[i, j] * 100)

labels.reverse()
top_confusions.reverse()

ax.barh(labels, top_confusions, color='#2b5c8f')
ax.set_xlabel('Misclassification Rate (%)', fontsize=12)
# ax.set_title('Top 12 Class Confusion Pairs (Quick, Draw! Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(chart_fname)
plt.close()

print(f"\nSaved confusion chart to {chart_fname}")