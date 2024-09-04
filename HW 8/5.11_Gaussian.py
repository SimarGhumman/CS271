import numpy as np
from scipy.stats import multivariate_normal

# Data from Table 5.6
data_points = np.array([
    [3.600, 79], [1.800, 54], [2.283, 62], [3.333, 74], [2.883, 55],
    [4.533, 85], [1.950, 51], [1.833, 54], [4.700, 88], [3.600, 85],
    [1.600, 52], [4.350, 85], [3.917, 84], [4.200, 78], [1.750, 62],
    [1.800, 51], [4.700, 83], [2.167, 52], [4.800, 84], [1.750, 47]
])

# Parameters from the first iteration (given from the user's image)
mu1 = np.array([2.6269, 63.0160])
S1 = np.array([[1.0548, 12.7306], [12.7306, 181.5183]])
mu2 = np.array([3.6756, 75.1981])
S2 = np.array([[1.2119, 14.1108], [14.1108, 189.2046]])
tau1 = 0.5704
tau2 = 0.4296

# Calculating the responsibilities (E-step)
p_ji = np.zeros((len(data_points), 2))

for i, point in enumerate(data_points):
    # Calculate the probability of each point under both Gaussian distributions
    p_x_given_theta1 = multivariate_normal.pdf(point, mean=mu1, cov=S1)
    p_x_given_theta2 = multivariate_normal.pdf(point, mean=mu2, cov=S2)

    # Calculate responsibilities
    p_ji[i, 0] = tau1 * p_x_given_theta1 / (tau1 * p_x_given_theta1 + tau2 * p_x_given_theta2)
    p_ji[i, 1] = tau2 * p_x_given_theta2 / (tau1 * p_x_given_theta1 + tau2 * p_x_given_theta2)

# Re-estimating the parameters (M-step)
mu1_new = np.dot(p_ji[:, 0], data_points) / np.sum(p_ji[:, 0])
mu2_new = np.dot(p_ji[:, 1], data_points) / np.sum(p_ji[:, 1])

S1_new = np.dot(p_ji[:, 0] * (data_points - mu1_new).T, (data_points - mu1_new)) / np.sum(p_ji[:, 0])
S2_new = np.dot(p_ji[:, 1] * (data_points - mu2_new).T, (data_points - mu2_new)) / np.sum(p_ji[:, 1])

tau1_new = np.sum(p_ji[:, 0]) / len(data_points)
tau2_new = np.sum(p_ji[:, 1]) / len(data_points)

print(p_ji)

print("Mean1: ", mu1_new, "S1: ", S1_new)
print("Mean1: ", mu2_new, "S1: ", S2_new)
print("T1: ", tau1_new, "T2: ", tau2_new)