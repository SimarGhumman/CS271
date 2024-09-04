import numpy as np

A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

B = np.array([[0.1, 0.4, 0.5],
              [0.7, 0.2, 0.1]])

pi = np.array([0.6, 0.4])


def compute_P_O_Q(sequence, states):
    prob = pi[states[0]] * B[states[0], sequence[0]]
    for i in range(1, len(states)):
        prob *= A[states[i - 1], states[i]] * B[states[i], sequence[i]]
    return prob


def forward_algorithm(sequence):
    alpha = np.zeros((len(sequence), 2))
    alpha[0, :] = pi * B[:, sequence[0]]

    for t in range(1, len(sequence)):
        for j in range(2):
            alpha[t, j] = np.dot(alpha[t - 1, :], A[:, j]) * B[j, sequence[t]]

    return alpha[-1, :].sum()


sequences = [(i, j, k, l) for i in range(3) for j in range(3) for k in range(3) for l in range(3)]
state_sequences = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

direct_probs = {}
forward_probs = {}

for seq in sequences:
    prob_direct = sum(compute_P_O_Q(seq, states) for states in state_sequences)
    prob_forward = forward_algorithm(seq)

    direct_probs[seq] = prob_direct
    forward_probs[seq] = prob_forward

print("\nDirect Calculation vs Forward Algorithm\n")
print(f"{'Sequence':20}{'Direct Calculation':25}{'Forward Algorithm'}")
print("-" * 60)
for seq in sequences:
    print(f"{str(seq):20}{direct_probs[seq]:<25.5f}{forward_probs[seq]:.5f}")

print("\nTotal Probability (Direct Calculation):", sum(direct_probs.values()))
print("Total Probability (Forward Algorithm):", sum(forward_probs.values()))
