import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Data from Table 5.6
old_faithful_data = np.array([
    [3.600, 79], [1.800, 54], [2.283, 62], [3.333, 74], [2.883, 55],
    [4.533, 85], [1.950, 51], [1.833, 54], [4.700, 88], [3.600, 85],
    [1.600, 52], [4.350, 85], [3.917, 84], [4.200, 78], [1.750, 62],
    [1.800, 51], [4.700, 83], [2.167, 52], [4.800, 84], [1.750, 47]
])

# Part a) Fitting GMM with initial means given in the problem statement
initial_means = np.array([[2.5, 65.0], [3.5, 70.0]])
gmm = GaussianMixture(n_components=2, means_init=initial_means, random_state=0)
gmm.fit(old_faithful_data)
cluster_assignments = gmm.predict(old_faithful_data)

# Output for part a
print("Part a\n")
for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
    print(f"Cluster{i + 1}:\n\tMean: {mean[0]:.8f}, {mean[1]:.8f}\n\tCovariance: [{cov[0, 0]:.8f}, {cov[0, 1]:.8f}], [{cov[1, 0]:.8f}, {cov[1, 1]:.8f}]\n")

# Part b) Choose different initial values than in part a
means_init_part_b = np.array([[2.0, 50.0], [4.0, 80.0]])
gmm_part_b = GaussianMixture(n_components=2, means_init=means_init_part_b, random_state=0)
gmm_part_b.fit(old_faithful_data)
cluster_assignments_part_b = gmm_part_b.predict(old_faithful_data)

# Output for part b
print("Part b\n")
for i, (mean, cov) in enumerate(zip(gmm_part_b.means_, gmm_part_b.covariances_)):
    print(f"Cluster{i + 1}:\n\tMean: {mean[0]:.8f}, {mean[1]:.8f}\n\tCovariance: [{cov[0, 0]:.8f}, {cov[0, 1]:.8f}], [{cov[1, 0]:.8f}, {cov[1, 1]:.8f}]\n")

# Part c) Use three clusters with arbitrary chosen initial values
means_init_part_c = np.array([[2.0, 60.0], [4.0, 80.0], [3.0, 70.0]])
gmm_part_c = GaussianMixture(n_components=3, means_init=means_init_part_c, random_state=0)
gmm_part_c.fit(old_faithful_data)
cluster_assignments_part_c = gmm_part_c.predict(old_faithful_data)

# Output for part c
print("Part c\n")
for i, (mean, cov) in enumerate(zip(gmm_part_c.means_, gmm_part_c.covariances_)):
    print(f"Cluster{i + 1}:\n\tMean: {mean[0]:.8f}, {mean[1]:.8f}\n\tCovariance: [{cov[0, 0]:.8f}, {cov[0, 1]:.8f}], [{cov[1, 0]:.8f}, {cov[1, 1]:.8f}]\n")

# Plotting the results of the EM algorithm for part a
em_labels_part_a = cluster_assignments  # Using cluster assignments from part a
plt.figure(figsize=(8, 6))
plt.scatter(old_faithful_data[:, 0], old_faithful_data[:, 1], c=em_labels_part_a, cmap='viridis', label='EM Clustering')
plt.title('EM Clustering of Old Faithful Data with K=2')
plt.xlabel('Eruption duration')
plt.ylabel('Waiting time')
plt.legend()
plt.show()
