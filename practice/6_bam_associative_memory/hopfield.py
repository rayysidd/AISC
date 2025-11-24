import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))  # Weight matrix

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.W += np.dot(p, p.T)  # Hebbian learning
        np.fill_diagonal(self.W, 0)  # No self-connections

    def recall(self, pattern, steps=5):
        x = pattern.copy()
        for _ in range(steps):
            x = np.sign(np.dot(self.W, x))
            # optional: handle zero -> +1
            x[x == 0] = 1
        return x


# Example binary patterns to store/recall
pattern1 = np.array([1, -1, 1, -1])
pattern2 = np.array([-1, 1, -1, 1])

patterns = [pattern1, pattern2]

hopfield = HopfieldNetwork(size=4)
hopfield.train(patterns)

# Test recall
test_pattern = pattern1.copy()
recalled_pattern = hopfield.recall(test_pattern)

print("Test Input:", test_pattern)
print("Recalled Output:", recalled_pattern)
