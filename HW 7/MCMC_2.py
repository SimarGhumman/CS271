import numpy as np

# Define the conditional probabilities for the proposal distribution
def proposal_prob(xt, Cx):
    if xt == 1:
        return 1 if Cx == 2 else 0
    elif xt == 6:
        return 1 if Cx == 5 else 0
    else:
        return 1/2 if Cx in [xt-1, xt+1] else 0

# Define the Metropolis-Hastings algorithm for the fair die problem
def metropolis_hastings_fair_die(iterations):
    counts = np.zeros(6)
    current_state = np.random.randint(1, 7)  # Start with a random state

    for _ in range(iterations):
        if current_state == 1:
            candidate = 2
        elif current_state == 6:
            candidate = 5
        else:
            candidate = current_state - 1 if np.random.rand() < 0.5 else current_state + 1

        u = np.random.uniform(0, 1)

        # Calculate the acceptance probability
        accept_prob = proposal_prob(current_state, candidate) / proposal_prob(candidate, current_state)

        # Accept or reject the candidate
        if u <= accept_prob:
            current_state = candidate

        # Count the occurrence
        counts[current_state - 1] += 1

    return counts

# Generate x_t for t = 1,2,...,1000
iterations = 1000
fair_die_counts = metropolis_hastings_fair_die(iterations)

# Display the counts for each value
print(fair_die_counts, fair_die_counts / iterations)