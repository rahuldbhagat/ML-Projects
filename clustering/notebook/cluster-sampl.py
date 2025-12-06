import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate synthetic 2D data (just for demo)
X, y_true = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Sample data")
plt.show()
