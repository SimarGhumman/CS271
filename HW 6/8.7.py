import math

class Data:
    def __init__(self, x0, x1, y):
        self.x0 = x0
        self.x1 = x1
        self.y = y

# Activation function and its derivative
def sigmoid(z):
    return (1 + math.exp(-z))

def sigmoid_derivative(z):
    return (-math.exp(-z))

# Initialize weights
w = [1, 2, -1, 1, -2, 1]

# Training using SGD
def train(training, epochs, alpha):
    global w
    for epoch in range(epochs):
        for data in training:
            # Forward pass
            v0 = w[0]
            v1 = w[1]
            v2 = w[2]
            v3 = w[3]
            v4 = w[4]
            v5 = w[5]
            v6 = data.x0 * v0 + data.x1 * v2
            v7 = data.x0 * v1 + data.x1 * v3
            v8 = sigmoid(v6)
            v9 = sigmoid(v7)
            v10 = v4 / v8
            v11 = v5 / v9
            v12 = ((v10 + v11 - data.y) ** 2) / 2
            z = v12

            # Backward pass
            dz = 1
            dv11 = v10 + v11 - data.y
            dv10 = v10 + v11 - data.y
            dv9 = (-v5 / (v9 ** 2)) * dv11
            dv8 = (-v4 / (v8 ** 2)) * dv10
            dv7 = sigmoid_derivative(v7) * dv9
            dv6 = sigmoid_derivative(v6) * dv8
            dv5 = dv11 / v9
            dv4 = dv10 / v8
            dv3 = data.x1 * dv7
            dv2 = data.x1 * dv6
            dv1 = data.x0 * dv7
            dv0 = data.x0 * dv6

            # Update weights using gradient descent
            w[0] -= alpha * dv0
            w[1] -= alpha * dv1
            w[2] -= alpha * dv2
            w[3] -= alpha * dv3
            w[4] -= alpha * dv4
            w[5] -= alpha * dv5

def classify(x0, x1):
    # Forward pass using trained weights
    v6 = x0 * w[0] + x1 * w[2]
    v7 = x0 * w[1] + x1 * w[3]
    v8 = sigmoid(v6)
    v9 = sigmoid(v7)
    v10 = w[4] / v8
    v11 = w[5] / v9

    return 1 if v10 + v11 > 0.5 else 0

def main():
    training = [
        Data(0.6, 0.4, 1),
        Data(0.1, 0.2, 0),
        Data(0.8, 0.6, 0),
        Data(0.3, 0.7, 1),
        Data(0.7, 0.3, 1),
        Data(0.7, 0.7, 0),
        Data(0.2, 0.9, 1)
    ]

    test = [
        Data(0.55, 0.11, 1),
        Data(0.32, 0.21, 0),
        Data(0.24, 0.64, 1),
        Data(0.86, 0.68, 0),
        Data(0.53, 0.79, 0),
        Data(0.46, 0.54, 1),
        Data(0.16, 0.51, 1),
        Data(0.52, 0.94, 0),
        Data(0.46, 0.87, 1),
        Data(0.96, 0.63, 0)
    ]

    # Train
    train(training, 1000, 0.1)

    # Evaluate
    correct_train = sum(1 for data in training if classify(data.x0, data.x1) == data.y)
    correct_test = sum(1 for data in test if classify(data.x0, data.x1) == data.y)

    print(f"Training accuracy: {100.0 * correct_train / len(training):.2f}%")
    print(f"Test accuracy: {100.0 * correct_test / len(test):.2f}%")
    print(f"Final weights: {' '.join(map(str, w))}")

if __name__ == "__main__":
    main()
