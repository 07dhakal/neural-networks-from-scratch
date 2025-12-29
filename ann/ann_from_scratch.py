import numpy as np

# ----------------------------------
# ANN from scratch for XOR problem
# ----------------------------------

class ANN:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.1):
        self.lr = lr

        # Weight initialization
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    # Sigmoid activation
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Derivative of sigmoid
    def sigmoid_derivative(self, a):
        return a * (1 - a)

    # Forward propagation
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    # Backward propagation
    def backward(self, X, y):
        m = X.shape[0]

        # Output layer error
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # Training loop
    def train(self, X, y, epochs=1000):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y)

            if i % 100 == 0:
                loss = np.mean((self.a2 - y) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

# -------------------------------
# Toy Dataset: XOR
# -------------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create model and train
model = ANN()
model.train(X, y, epochs=1000)

# Predictions after training
print("Predictions after training:")
print(model.forward(X))
