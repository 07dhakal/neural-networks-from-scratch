import numpy as np

# ---------------------------------
# Simple CNN forward pass only
# ---------------------------------
class SimpleCNN:
    def __init__(self):
        # 3x3 convolution filter
        self.filter = np.random.randn(3, 3)
        # Fully connected layer weights
        self.fc = np.random.randn(4, 1)

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
        conv = self.convolve(image)
        flat = conv.flatten().reshape(-1, 1)
        output = flat.T @ self.fc
        return output

# Toy image dataset
image = np.array([
    [1,0,1,0,1],
    [0,1,0,1,0],
    [1,0,1,0,1],
    [0,1,0,1,0],
    [1,0,1,0,1]
])

cnn = SimpleCNN()
print("CNN Forward Output:", cnn.forward(image))
