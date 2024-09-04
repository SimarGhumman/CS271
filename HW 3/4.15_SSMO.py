import numpy as np
import random
import matplotlib.pyplot as plt

X = np.array([[3,3], [3,4], [2,3], [1,1], [1,3], [2,2]])
z = np.array([1, 1, 1, -1, -1, -1])

lambdas = np.zeros(len(z))
b = 0

C = 2.5
epsilon = 0.00001

def kernel(x1, x2):
    return np.dot(x1, x2)

pairs = [(0,1), (1,2), (2,3), (3,4), (4,5),
         (0,2), (1,3), (2,4), (3,5),
         (0,3), (1,4), (2,5),
         (0,4), (1,5),
         (0,5),
         (1,0), (2,1), (3,2), (4,3), (5,4),
         (2,0), (3,1), (4,2), (5,3),
         (3,0), (4,1), (5,2),
         (4,0), (5,1),
         (5,0)]


def generate_pairs(num_pairs=1000):
    pairs = []
    i_values = list(range(6))

    for _ in range(num_pairs):
        i = random.choice(i_values)
        possible_j_values = [j for j in i_values if j != i]
        j = random.choice(possible_j_values)
        pairs.append((i, j))

    return pairs

def compute_E(i, lambdas, b, X, z, kernel):
    f_xi = np.sum(lambdas * z * [kernel(x, X[i]) for x in X]) + b
    E_i = f_xi - z[i]
    return E_i

def SSMO(X, z, lambdas, b, C, epsilon, kernel, pairs, max_passes=10):
    passes = 0
    while passes < max_passes:
        num_changed = 0
        for i, j in pairs:
            d = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
            if d < -epsilon:
                E_i = compute_E(i, lambdas, b, X, z, kernel)
                E_j = compute_E(j, lambdas, b, X, z, kernel)

                lam_i_old = lambdas[i]
                lam_j_old = lambdas[j]

                lambdas[j] -= z[j] * (E_i - E_j) / d

                if z[i] == z[j]:
                    L = max(0, lambdas[i] + lambdas[j] - C)
                    H = min(C, lambdas[i] + lambdas[j])
                else:
                    L = max(0, lambdas[j] - lambdas[i])
                    H = min(C, C + lambdas[j] - lambdas[i])

                if lambdas[j] > H:
                    lambdas[j] = H
                elif L <= lambdas[j] <= H:
                    lambdas[j] = lambdas[j]
                else:
                    lambdas[j] = L

                lambdas[i] += z[i] * z[j] * (lam_j_old - lambdas[j])

                bi = b - E_i - z[i] * (lambdas[i] - lam_i_old) * kernel(X[i], X[i]) - z[j] * (lambdas[j] - lam_j_old) * kernel(X[i], X[j])
                bj = b - E_j - z[i] * (lambdas[i] - lam_i_old) * kernel(X[i], X[j]) - z[j] * (lambdas[j] - lam_j_old) * kernel(X[j], X[j])

                if 0 < lambdas[i] < C:
                    b = bi
                elif 0 < lambdas[j] < C:
                    b = bj
                else:
                    b = (bi + bj) / 2

                num_changed += 1

        if num_changed == 0:
            return lambdas, b
        else:
            passes += 1

    return lambdas, b


def plot_hyperplane(X, z, lambdas, b, title):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[z == 1][:, 0], X[z == 1][:, 1], c='blue', marker='o', label='Positive Class (+1)')
    plt.scatter(X[z == -1][:, 0], X[z == -1][:, 1], c='red', marker='x', label='Negative Class (-1)')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([np.sum(lambdas * z * [kernel(x, x_i) for x_i in X]) for x in Z]) + b
    predictions = predictions.reshape(xx.shape)

    plt.contour(xx, yy, predictions, levels=[0], colors='green', linewidths=2)

    plt.title(title)
    plt.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

lambdas, b = SSMO(X, z, lambdas, b, C, epsilon, kernel, pairs)

print("Part A")
print(lambdas, b)
plot_hyperplane(X, z, lambdas, b, "Hyperplane for Part a)")

lambdas = np.zeros(len(z))
b = 0

new_pairs = generate_pairs()
lambdas, b = SSMO(X, z, lambdas, b, C, epsilon, kernel, new_pairs)

print("Part B")
print(lambdas, b)
plot_hyperplane(X, z, lambdas, b, "Hyperplane for Part b)")