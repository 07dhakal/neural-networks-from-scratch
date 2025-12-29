import numpy as np

# ---------------------------------
# Simple Vanilla RNN forward pass
# ---------------------------------
class SimpleRNN:
    def __init__(self, input_size=1, hidden_size=2, output_size=1):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size)  # input → hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) # hidden → hidden
        self.Why = np.random.randn(output_size, hidden_size) # hidden → output
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    # Forward pass through time
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        for x in inputs:
            x = np.array([[x]])
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        y = self.Why @ h + self.by
        return y

# Toy sequence
sequence = [1, 2, 3]

rnn = SimpleRNN()
print("RNN Forward Output:", rnn.forward(sequence))
