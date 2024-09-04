import numpy as np


class HMM:
    def __init__(self, unique_chars):
        self.N = 2
        self.M = 27

        self.char_to_index = {char: index for index, char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index, char in enumerate(unique_chars)}

        perturbation_pi = 0.02
        self.pi = np.array([0.5 + np.random.uniform(-perturbation_pi, perturbation_pi) for _ in range(self.N)])
        self.pi /= np.sum(self.pi)

        perturbation_a = 0.02
        self.A = np.array(
            [[0.5 + np.random.uniform(-perturbation_a, perturbation_a) for _ in range(self.N)] for _ in range(self.N)])
        self.A /= np.sum(self.A, axis=1).reshape(-1, 1)

        perturbation_b = 0.005
        even_value = 1.0 / self.M
        self.B = np.full((self.N, self.M), even_value)
        self.B += np.random.uniform(-perturbation_b, perturbation_b, self.B.shape)
        self.B /= np.sum(self.B, axis=1).reshape(-1, 1)

    def tokenize(self, text):
        return np.array([self.char_to_index[char] for char in text if char in self.char_to_index])

    def detokenize(self, tokens):
        return ''.join([self.index_to_char[index] for index in tokens])

    def generate_sequence(self, length):
        state_sequence = [np.random.choice(self.N, p=self.pi)]
        observation_sequence = [np.random.choice(self.M, p=self.B[state_sequence[0]])]

        for _ in range(1, length):
            next_state = np.random.choice(self.N, p=self.A[state_sequence[-1]])
            next_observation = np.random.choice(self.M, p=self.B[next_state])

            state_sequence.append(next_state)
            observation_sequence.append(next_observation)

        return self.detokenize(observation_sequence)

    def train(self, observations, maxIters=100):
        T = len(observations)
        c = np.zeros(T)

        oldLogProb = -np.inf
        iters = 0

        while True:
            alpha = np.zeros((T, self.N))

            c[0] = 0
            for i in range(self.N):
                alpha[0, i] = self.pi[i] * self.B[i, observations[0]]
                c[0] += alpha[0, i]

            c[0] = 1 / c[0]
            alpha[0] *= c[0]

            for t in range(1, T):
                c[t] = 0
                for i in range(self.N):
                    for j in range(self.N):
                        alpha[t, i] += alpha[t - 1, j] * self.A[j, i]
                    alpha[t, i] *= self.B[i, observations[t]]
                    c[t] += alpha[t, i]

                c[t] = 1 / c[t]
                alpha[t] *= c[t]

            beta = np.zeros((T, self.N))

            beta[-1] = c[-1]

            for t in range(T - 2, -1, -1):
                for i in range(self.N):
                    for j in range(self.N):
                        beta[t, i] += self.A[i, j] * self.B[j, observations[t + 1]] * beta[t + 1, j]

                    beta[t, i] *= c[t]

            gamma = np.zeros((T, self.N, self.N))
            gamma_i = np.zeros((T, self.N))

            for t in range(T - 1):
                for i in range(self.N):
                    gamma_i[t, i] = 0
                    for j in range(self.N):
                        gamma[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, observations[t + 1]] * beta[t + 1, j]
                        gamma_i[t, i] += gamma[t, i, j]

            gamma_i[-1] = alpha[-1]

            self.pi = gamma_i[0] / np.sum(gamma_i[0])

            for i in range(self.N):
                denom = np.sum(gamma_i[:-1, i])
                for j in range(self.N):
                    self.A[i, j] = np.sum(gamma[:-1, i, j]) / denom

                denom = np.sum(gamma_i[:, i])
                for j in range(self.M):
                    self.B[i, j] = np.sum(gamma_i[observations == j, i]) / denom
                self.B[i, :] /= np.sum(self.B[i, :])

            logProb = -np.sum(np.log(c))

            iters += 1
            if iters >= maxIters or logProb <= oldLogProb:
                break

            oldLogProb = logProb
        return logProb


def preprocess(text):
    text = text.lower()
    return ''.join([c for c in text if c in "abcdefghijklmnopqrstuvwxyz "])

with open("brown.txt", "r") as f:
    text = preprocess(f.read())[:50000]

unique_chars = list("abcdefghijklmnopqrstuvwxyz ")

model = HMM(unique_chars)

print(f"N: {model.N}")
print(f"M: {model.M}")
print(f"T: {len(text)}")
print("\nInitial π:")
print(model.pi)
print("\nInitial Transition Matrix A:")
print(model.A)
print("\nInitial Emission Matrix B:")
print(model.B)

print("\nSum of π values:", np.sum(model.pi))
print("Sum of rows in A:", np.sum(model.A, axis=1))
print("Sum of rows in B:", np.sum(model.B, axis=1))

tokenized_text = model.tokenize(text)

logProb = model.train(tokenized_text, maxIters=200)

np.set_printoptions(suppress=True, precision=10)

print("\nFinal π:")
print(model.pi)
print("\nFinal Transition Matrix A:")
print(model.A)
print("\nFinal Emission Matrix B:")
for char, index in model.char_to_index.items():
    state1_prob = round(model.B[0, index], 6)
    state2_prob = round(model.B[1, index], 6)
    print(f"{char}: {state1_prob} {state2_prob}")
print("Sum of rows in B:", np.sum(model.B, axis=1))

print(f"log[P(O | λ)] after training: {logProb}")
