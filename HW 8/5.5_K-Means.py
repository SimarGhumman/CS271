import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Old Faithful eruption data from the image provided
old_faithful_data = [
    [1, 3.600, 79], [2, 1.800, 54], [3, 2.283, 62], [4, 3.333, 74], [5, 2.883, 55],
    [6, 4.533, 85], [7, 1.950, 51], [8, 1.833, 54], [9, 4.700, 88], [10, 3.600, 85],
    [11, 1.600, 52], [12, 4.350, 85], [13, 3.917, 84], [14, 4.200, 78], [15, 1.750, 62],
    [16, 1.800, 51], [17, 4.700, 83], [18, 2.167, 52], [19, 4.800, 84], [20, 1.750, 47]
]

# Converting the data to a numpy array, excluding the case number
data = np.array(old_faithful_data)[:, 1:]

# Implementing K-means clustering for K=2 and K=3
k_values = [2, 3]
clusters = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    clusters[k] = kmeans.labels_

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Clustering of Old Faithful Eruption Data')

for i, k in enumerate(k_values):
    axs[i].scatter(data[:, 0], data[:, 1], c=clusters[k], cmap='viridis', marker='o')
    axs[i].set_title(f'K-Means with K={k}')
    axs[i].set_xlabel('Eruption duration')
    axs[i].set_ylabel('Waiting time')

plt.show()

# Returning the cluster labels for K=2 and K=3
print(clusters[2], clusters[3])