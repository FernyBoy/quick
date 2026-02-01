import h5py
import numpy as np
import matplotlib.pyplot as plt


def visualize_confused_pairs(hdf5_path, label_names, pair=(21, 39), num_samples=5):
    """
    Displays samples of two classes that the model frequently confuses.
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images']
        labels = f['labels']

        # Get class names if available, else use indices
        class_a_name = label_names[pair[0]]
        class_b_name = label_names[pair[1]]

        # Find indices where the labels match our target pair
        # We search a large chunk to find samples quickly
        search_limit = 50000
        subset_labels = labels[:search_limit]

        idx_a = np.where(subset_labels == pair[0])[0]
        idx_b = np.where(subset_labels == pair[1])[0]
        np.random.shuffle(idx_a)
        np.random.shuffle(idx_b)
        for j in range(5):  # Shuffle multiple times for better randomness
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
            fig.suptitle(
                f'Visual Similarity: {class_a_name} (Top) vs {class_b_name} (Bottom)',
                fontsize=16,
            )

            for k in range(num_samples):
                i = num_samples * j + k
                # Plot Class A
                axes[0, k].imshow(images[idx_a[i]].reshape(28, 28), cmap='gray_r')
                axes[0, k].axis('off')
                # Plot Class B
                axes[1, k].imshow(images[idx_b[i]].reshape(28, 28), cmap='gray_r')
                axes[1, k].axis('off')

            plt.tight_layout()
            plt.show()


# Run for your top two confusion pairs
# Update 'path_to_your_data.h5' with your actual filename
h5_path = 'data/quick/prep_dataset.h5'
csv_path = 'data/quick/prep_names.csv'

with open(csv_path, 'r') as file:
    # Read all lines into a list, stripping the newline character from each line
    label_names = [line.strip() for line in file]

print('Visualizing Pair 21 vs 39...')
visualize_confused_pairs(h5_path, label_names, pair=(21, 39))

print('Visualizing Pair 54 vs 59...')
visualize_confused_pairs(h5_path, label_names, pair=(54, 59))
