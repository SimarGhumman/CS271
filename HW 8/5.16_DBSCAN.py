from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Dataset provided
X = np.array([
    [1.0,5.0], [1.25,5.35], [1.25,5.75], [1.5,6.25], [1.75,6.75], [2.0,6.5], [3.0,7.75], [3.5,8.25], [3.75,8.75],
    [3.95,9.1], [4.0,8.5], [2.5,7.25], [2.25,7.75], [2.0,6.5], [2.75,8.25], [4.5,8.9], [9.0,5.0], [8.75,5.85],
    [9.0,6.25], [8.0,7.0], [8.5,6.25], [8.5,6.75], [8.25,7.65], [7.0,8.25], [6.0,8.75], [5.5,8.25], [5.25,8.75],
    [4.9,8.75], [5.0,8.5], [7.5,7.75], [7.75,8.25], [6.75,8.0], [6.25,8.25], [4.5,8.9], [5.0,1.0], [1.25,4.65],
    [1.25,4.25], [1.5,3.75], [1.75,3.25], [2.0,3.5], [3.0,2.25], [3.5,1.75], [3.75,8.75], [3.95,0.9], [4.0,1.5],
    [2.5,2.75], [2.25,2.25], [2.0,3.5], [2.75,1.75], [4.5,1.1], [5.0,9.0], [8.75,5.15], [8.0,2.25], [8.25,3.0],
    [8.5,4.75], [8.5,4.25], [8.25,3.35], [7.0,1.75], [8.0,3.5], [6.0,1.25], [5.5,1.75], [5.25,1.25], [4.9,1.25],
    [5.0,1.5], [7.5,2.25], [7.75,2.75], [6.75,2.0], [6.25,1.75], [4.5,1.1], [3.0,4.5], [7.0,4.5], [5.0,3.0],
    [4.0,3.35], [6.0,3.35], [4.25,3.25], [5.75,3.25], [3.5,3.75], [6.5,3.75], [3.25,4.0], [6.75,4.0], [3.75,3.55],
    [6.25,3.55], [4.75,3.05], [5.25,3.05], [4.5,3.15], [5.5,3.15], [4.0,6.5], [4.0,6.75], [4.0,6.25], [3.75,6.5],
    [4.25,6.5], [4.25,6.75], [3.75,6.25], [6.0,6.5], [6.0,6.75], [6.0,6.25], [5.75,6.75], [5.75,6.25], [6.25,6.75],
    [6.25,6.25], [9.5,9.5], [2.5,9.5], [1.0,8.0]
])

# Parameters to test
params = [(0.6, 3), (0.75, 4), (1.0, 5), (2.0, 10)]

# Iterate over each pair of parameters and apply DBSCAN
for eps, min_samples in params:
    # Apply DBSCAN to the data
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Extract labels
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Plotting
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        # Plot points for each cluster
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    plt.title(f'DBSCAN clustering with eps={eps}, min_samples={min_samples}\nEstimated number of clusters: {n_clusters}, Noise points: {n_noise}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()