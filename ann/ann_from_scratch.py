import numpy as np

class ANN:
    def __init__(self):
        self.W = np.random.randn(2, 1)
        self.b = np.zeros((1,))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

if __name__ == "__main__":
    model = ANN()
    X = np.array([[1, 2]])
    print(model.forward(X))
