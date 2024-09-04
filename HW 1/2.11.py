import numpy as np


def caesar_cipher(text, shift):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    mapping = str.maketrans(alphabet, shifted_alphabet)
    return text.translate(mapping)


def create_digraph_frequency_matrix(text):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    matrix = np.zeros((26, 26), dtype=int)

    for i in range(len(text) - 1):
        x = alphabet.index(text[i])
        y = alphabet.index(text[i + 1])
        matrix[x][y] += 1

    matrix += 5
    matrix = matrix.astype(float)
    matrix /= np.sum(matrix, axis=1).reshape(-1, 1)

    return matrix

class HMMFixedA:
    def __init__(self, unique_chars, transition_matrix):
        self.N = 26
        self.M = 26
        self.char_to_index = {char: index for index, char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index, char in enumerate(unique_chars)}

        perturbation_pi = 0.02
        self.pi = np.array([0.5 + np.random.uniform(-perturbation_pi, perturbation_pi) for _ in range(self.N)])
        self.pi /= np.sum(self.pi)
        self.A = transition_matrix

        perturbation_b = 0.005
        even_value = 1.0 / self.M
        self.B = np.full((self.N, self.M), even_value)
        self.B += np.random.uniform(-perturbation_b, perturbation_b, self.B.shape)
        self.B /= np.sum(self.B, axis=1).reshape(-1, 1)


    def tokenize(self, text):
            return np.array([self.char_to_index[char] for char in text if char in self.char_to_index])

    def detokenize(self, tokens):
        return ''.join([self.index_to_char[index] for index in tokens])

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
                    alpha[t, i] = np.sum([alpha[t - 1, j] * self.A[j, i] for j in range(self.N)]) * self.B[i, observations[t]]
                    c[t] += alpha[t, i]
                c[t] = 1 / c[t]
                alpha[t] *= c[t]

            beta = np.zeros((T, self.N))
            beta[-1] = c[-1]

            for t in range(T - 2, -1, -1):
                for i in range(self.N):
                    beta[t, i] = np.sum([self.A[i, j] * self.B[j, observations[t + 1]] * beta[t + 1, j] for j in range(self.N)])
                    beta[t, i] *= c[t]

            gamma = np.zeros((T, self.N))
            for t in range(T):
                for i in range(self.N):
                    gamma[t, i] = alpha[t, i] * beta[t, i] / np.sum([alpha[t, j] * beta[t, j] for j in range(self.N)])

            for i in range(self.N):
                for j in range(self.M):
                    self.B[i, j] = np.sum([gamma[t, i] for t in range(T) if observations[t] == j]) / np.sum(gamma[:, i])

            logProb = -np.sum(np.log(c))

            iters += 1
            if iters >= maxIters or logProb <= oldLogProb:
                break

            oldLogProb = logProb
        return logProb

class HMM:
    def __init__(self, unique_chars):
        self.N = 2
        self.M = 27

        self.char_to_index = {char: index for index, char in enumerate(unique_chars)}
        self.index_to_char = {index: char for index, char in enumerate(unique_chars)}

        perturbation_pi = 0.02
        self.pi = np.array([0.5 + np.random.uniform(-perturbation_pi, perturbation_pi),
                            0.5 + np.random.uniform(-perturbation_pi, perturbation_pi)])
        self.pi /= np.sum(self.pi)

        perturbation_a = 0.02
        self.A = np.array([[0.5 + np.random.uniform(-perturbation_a, perturbation_a),
                            0.5 + np.random.uniform(-perturbation_a, perturbation_a)],
                           [0.5 + np.random.uniform(-perturbation_a, perturbation_a),
                            0.5 + np.random.uniform(-perturbation_a, perturbation_a)]])
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
    return ''.join([c for c in text if c.isalpha()])

with open("brown.txt", "r") as f:
    text = preprocess(f.read())[:50000]

shift = np.random.randint(1, 26)
encrypted_text = caesar_cipher(text, shift)
print(f"Random Shift Used for Encryption: {shift}")

unique_chars = list("abcdefghijklmnopqrstuvwxyz")
model = HMM(unique_chars)

tokenized_encrypted_text = model.tokenize(encrypted_text)

logProb = model.train(tokenized_encrypted_text, maxIters=200)

with open("brown.txt", "r") as f:
    long_text = preprocess(f.read())[:1000000]
matrix_A = create_digraph_frequency_matrix(long_text)

model.A = matrix_A

print("\nFinal Emission Matrix B:")
for char, index in model.char_to_index.items():
    state1_prob = round(model.B[0, index], 6)
    state2_prob = round(model.B[1, index], 6)
    print(f"{char}: {state1_prob} {state2_prob}")

short_encrypted_text = encrypted_text[:1000]

model_d = HMMFixedA(unique_chars, matrix_A)

tokenized_short_encrypted_text = model_d.tokenize(short_encrypted_text)

logProb_d = model_d.train(tokenized_short_encrypted_text, maxIters=200)

putative_key_d = np.argmax(model_d.B, axis=1)

actual_key = [shift for _ in range(26)]
matches_d = np.sum(putative_key_d == actual_key)
fraction_d = matches_d / 26

print(f"Fraction of putative key elements (for first 1000 characters) that match the actual key: {fraction_d:.4f}")