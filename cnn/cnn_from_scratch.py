import numpy as np

# ---------------------------------
# Simple CNN with toy backward pass
# ---------------------------------
class SimpleCNN:
    def __init__(self, lr=0.01):
        # 3x3 convolution filter
        self.filter = np.random.randn(3, 3)
        # Fully connected layer weights (4 inputs → 1 output)
        self.fc = np.random.randn(4, 1)
        self.lr = lr

    # Manual convolution
    def convolve(self, image):
        output = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                region = image[i:i+3, j:j+3]
                output[i, j] = np.sum(region * self.filter)
        return output

    # Forward pass
    def forward(self, image):
        self.conv_out = self.convolve(image)
        self.flat = self.conv_out.flatten().reshape(-1, 1)
        self.output = self.flat.T @ self.fc
        return self.output

    # Toy backward pass (only updates FC weights)
    def backward(self, target):
        error = self.output - target
        grad_fc = self.flat * error
        self.fc -= self.lr * grad_fc
        return np.mean(error**2)

    # Training loop
    def train(self, image, target, epochs=100):
        for i in range(epochs):
            self.forward(image)
            loss = self.backward(target)
            if i % 20 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

# -------------------------------
# Toy image dataset
# -------------------------------
image = np.array([
    [1,0,1,0,1],
    [0,1,0,1,0],
    [1,0,1,0,1],
    [0,1,0,1,0],
    [1,0,1,0,1]
])

target = np.array([[1]])  # arbitrary target for demonstration

cnn = SimpleCNN()
cnn.train(image, target, epochs=100)

print("CNN Output after training:", cnn.forward(image))

