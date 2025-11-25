import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw >= 0 else -1
        return pattern

# Create 9x9 binary pattern for a Devanagari character (example character made-up pattern)
char_A = np.array([
    [0,0,1,1,1,1,1,0,0],
    [0,1,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,1,0],
    [0,0,1,1,1,1,1,0,0]
])

def binarize(pattern):
    return np.where(pattern == 0, -1, 1).flatten()

patterns = [binarize(char_A)]
size = 81

hopfield = HopfieldNetwork(size)
hopfield.train(patterns)

# Test with noisy version (flip a bit)
noisy = patterns[0].copy()
noisy[0] = -noisy[0]

recalled = hopfield.recall(noisy, steps=10)

print("Original Pattern:\n", patterns[0].reshape(9,9))
print("Noisy Input:\n", noisy.reshape(9,9))
print("Recalled Pattern:\n", recalled.reshape(9,9))