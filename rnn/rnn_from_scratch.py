import numpy as np

# ---------------------------------
# Simple RNN with toy backward
# ---------------------------------
class SimpleRNN:
    def __init__(self, input_size=1, hidden_size=2, output_size=1, lr=0.01):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size)   # input → hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)  # hidden → hidden
        self.Why = np.random.randn(output_size, hidden_size)  # hidden → output
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.lr = lr

    # Forward pass
    def forward(self, inputs):
        self.h = np.zeros((self.hidden_size, 1))
        for x in inputs:
            x = np.array([[x]])
            self.h = np.tanh(self.Wxh @ x + self.Whh @ self.h + self.bh)
        self.y = self.Why @ self.h + self.by
        return self.y

    # Toy backward pass (update only output weights)
    def backward(self, target):
        error = self.y - target
        grad_Why = error @ self.h.T
        grad_by = error
        self.Why -= self.lr * grad_Why
        self.by -= self.lr * grad_by
        return np.mean(error**2)

    # Training loop
    def train(self, inputs, target, epochs=100):
        for i in range(epochs):
            self.forward(inputs)
            loss = self.backward(target)
            if i % 20 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

# -------------------------------
# Toy sequence dataset
# -------------------------------
sequence = [1, 2, 3]
target = np.array([[1]])  # arbitrary target

rnn = SimpleRNN()
rnn.train(sequence, target, epochs=100)

print("RNN Output after training:", rnn.forward(sequence))
