import numpy as np

class BAM:
    def __init__(self, input_len, output_len):
        self.W = np.zeros((input_len, output_len))

    def train(self, input_vectors, output_vectors):
        for x, y in zip(input_vectors, output_vectors):
            self.W += np.outer(x, y)

    def recall(self, input_pattern, steps=5):
        x = input_pattern.copy()
        for _ in range(steps):
            y = np.sign(np.dot(x, self.W))
            x = np.sign(np.dot(self.W, y))
        return y

# Example input/output binary vectors representing related/unrelated pairs:
# Dog (input) and Chair (output)
dog = np.array([1, -1, 1, -1])
chair = np.array([-1, 1, -1, 1])

# Cat (input) and Chair (output)
cat = np.array([-1, 1, -1, 1])

input_vectors = [dog, cat]
output_vectors = [chair, chair]

bam = BAM(input_len=4, output_len=4)
bam.train(input_vectors, output_vectors)

# Recall
test_input = dog
recalled_output = bam.recall(test_input)

print("Input:", test_input)
print("Recalled Output:", recalled_output)