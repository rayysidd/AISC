import numpy as np

class neuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, mode='binary'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.mode = mode

        np.random.seed(42)

        # FIX 1 — Weight matrices must have correct shapes
        self.W1 = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.B1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
        self.B2 = np.zeros((1, self.output_size))

    def activation(self, x):
        if self.mode == 'binary':
            return 1 / (1 + np.exp(-x))     # Sigmoid
        else:
            return (2 / (1 + np.exp(-x))) - 1   # Bipolar sigmoid
        
    def derivative(self, f_x):
        if self.mode == 'binary':
            return f_x * (1 - f_x)   # Sigmoid derivative
        else:
            return 0.5 * (1 + f_x) * (1 - f_x)  # Bipolar sigmoid derivative
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.B1
        self.a1 = self.activation(self.z1)
       
        self.z2 = np.dot(self.a1, self.W2) + self.B2
        self.a2 = self.activation(self.z2)
        return self.a2
    
    def train(self, X, y, epochs):

        for epoch in range(epochs):

            # Forward pass
            output = self.forward(X)

            # Error
            error = y - output

            # FIX 3 — No LR inside delta
            delta_output = error * self.derivative(output)
            
            # Backprop into hidden layer
            error_hidden = delta_output.dot(self.W2.T)

            # FIX 2 — Use derivative, not activation
            delta_hidden = error_hidden * self.derivative(self.a1)

            # Weight updates
            self.W2 += self.a1.T.dot(delta_output) * self.lr
            self.B2 += np.sum(delta_output, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(delta_hidden) * self.lr
            self.B1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

            # Monitoring
            if epoch % 10 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch}, MSE Loss: {loss:.5f}")

    def predict(self, X):
        out = self.forward(X)
        if self.mode == 'binary':
            return (out > 0.5).astype(int)
        else:
            return np.where(out >= 0, 1, -1)

if __name__ == "__main__":
    # XOR dataset (binary mode)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])

    nn = neuralNet(input_size=2, hidden_size=2, output_size=1, lr=0.5, mode='binary')
    nn.train(X, y, epochs=10000)

    preds = nn.predict(X)
    outputs = nn.forward(X)

    print("\nInputs:\n", X)
    print("Targets:\n", y)
    print("Raw outputs:\n", np.round(outputs, 5))
    print("Predictions:\n", preds)
    print("Accuracy:", np.mean(preds == y))