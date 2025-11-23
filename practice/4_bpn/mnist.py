import numpy as np

class MNIST_Backprop:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, lr=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        # Weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.B1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.B2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y_onehot, epochs=100):
        # y_onehot shape should be (N, 10)
        for i in range(epochs):
            # Forward
            z1 = np.dot(X, self.W1) + self.B1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.B2
            a2 = self.sigmoid(z2)

            # Error
            error = y_onehot - a2
            
            # Backward
            d_output = error * self.sigmoid_derivative(a2)
            d_hidden = d_output.dot(self.W2.T) * self.sigmoid_derivative(a1)
            
            # Update
            self.W2 += a1.T.dot(d_output) * self.lr
            self.B2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_hidden) * self.lr
            self.B1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr
            
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {np.mean(np.abs(error)):.4f}")

# Usage Hint for B3:
# You must One-Hot Encode your labels!
# Label '2' becomes [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]