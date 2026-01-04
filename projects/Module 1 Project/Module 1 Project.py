import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import log10

def load_mnist(path, kind='train'):
    """Load uncompressed MNIST data from `path`."""
    if kind == 'train':
        labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
        images_path = os.path.join(path, 'train-images.idx3-ubyte')
    else:  # 't10k' for test
        labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
        images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    
    print(f"Attempting to open labels: {labels_path}")
    try:
        with open(labels_path, 'rb') as lbpath:
            struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    except Exception as e:
        print(f"Error opening labels file: {e}")
        raise
    
    print(f"Attempting to open images: {images_path}")
    try:
        with open(images_path, 'rb') as imgpath:
            struct.unpack('>IIII', imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    except Exception as e:
        print(f"Error opening images file: {e}")
        raise
    
    return images, labels

# Load the data
data_dir = r'mnist_data'  
print(f"Data directory: {os.path.abspath(data_dir)}")

X_train, y_train = load_mnist(data_dir, kind='train')
X_test, y_test = load_mnist(data_dir, kind='t10k')

# Use first 2000 samples from test set, scale to [0,1]
X = X_test[:2000] / 255.0
y = y_test[:2000]

print(f"Data shape: {X.shape}")

# Center the data
mean_X = np.mean(X, axis=0)
X_centered = X - mean_X

# Compute sample covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Percentage of variance explained
variance_explained = (eigenvalues / np.sum(eigenvalues)) * 100
cumulative_variance = np.cumsum(variance_explained)

# Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('Cumulative Variance Explained by Principal Components')
plt.grid(True)
plt.show()

n_90 = np.argmax(cumulative_variance >= 90) + 1
print(f"Components needed for 90% variance: {n_90}")

# Dimensionality reduction and reconstruction
reduced_data = {}
reconstructed = {}

for p in [50, 250, 500]:
    V_p = eigenvectors[:, :p]
    X_reduced = X_centered @ V_p
    reduced_data[p] = X_reduced
    X_rec = (X_reduced @ V_p.T) + mean_X
    reconstructed[p] = X_rec
    print(f"Reconstructed shape for p={p}: {X_rec.shape}")

# Visualize one example
def show_image(original, recon, title):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original.reshape(28, 28), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(recon.reshape(28, 28), cmap='gray')
    axs[1].set_title(title)
    plt.show()

idx = 0
for p in [50, 250, 500]:
    show_image(X[idx], reconstructed[p][idx], f'Reconstructed (p={p})')

# PSNR
def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10(1 / mse)

np.random.seed(42)
random_indices = np.random.choice(range(2000), 5, replace=False)

psnr_results = {p: [] for p in [50, 250, 500]}

for idx in random_indices:
    original = X[idx]
    print(f"\nImage {idx} (label: {y[idx]}):")
    for p in [50, 250, 500]:
        recon = reconstructed[p][idx]
        psnr = compute_psnr(original, recon)
        psnr_results[p].append(psnr)
        print(f"  PSNR for p={p}: {psnr:.2f} dB")

for p in [50, 250, 500]:
    avg_psnr = np.mean(psnr_results[p])
    print(f"\nAverage PSNR for p={p}: {avg_psnr:.2f} dB")