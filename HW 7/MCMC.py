import numpy as np

# Define the probability distribution of the loaded die
prob_dist = np.array([1/10]*5 + [1/2])

# Define the Metropolis algorithm for the loaded die
def metropolis_algorithm(prob_dist, iterations):
    counts = np.zeros_like(prob_dist)
    current_state = np.random.choice(np.arange(1, 7), p=prob_dist)

    for _ in range(iterations):
        candidate = np.random.randint(1, 7)  # Roll a fair die
        u = np.random.uniform(0, 1)  # Generate a uniform random number

        # Calculate the acceptance probability
        accept_prob = prob_dist[candidate - 1] / prob_dist[current_state - 1]

        # Accept or reject the candidate
        if u <= accept_prob:
            current_state = candidate

        # Count the occurrence
        counts[current_state - 1] += 1

    return counts

# Generate x_t for t = 1,2,...,1000
iterations = 1000
counts_1 = metropolis_algorithm(prob_dist, iterations)

prob_dist_2 = np.array([3/4] + [1/20]*5)
counts_2 = metropolis_algorithm(prob_dist_2, iterations)

# Display the counts for each side of the die
print(counts_1, counts_1 / iterations)
print(counts_2, counts_2 / iterations)