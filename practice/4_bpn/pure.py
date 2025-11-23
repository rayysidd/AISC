import math
import random

class PythonNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = 0.5
        
        # Init weights matrix manually
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, inputs):
        # Layer 1
        self.hidden_outputs = []
        for j in range(self.hidden_size):
            activation = sum(inputs[i] * self.W1[i][j] for i in range(self.input_size))
            self.hidden_outputs.append(self.sigmoid(activation))
            
        # Layer 2
        self.final_outputs = []
        for k in range(self.output_size):
            activation = sum(self.hidden_outputs[j] * self.W2[j][k] for j in range(self.hidden_size))
            self.final_outputs.append(self.sigmoid(activation))
        return self.final_outputs

    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            outputs = self.predict(inputs)
            
            # Calculate Output Deltas
            output_deltas = []
            for k in range(self.output_size):
                error = targets[k] - outputs[k]
                output_deltas.append(error * outputs[k] * (1 - outputs[k]))

            # Calculate Hidden Deltas
            hidden_deltas = []
            for j in range(self.hidden_size):
                error = sum(output_deltas[k] * self.W2[j][k] for k in range(self.output_size))
                hidden_deltas.append(error * self.hidden_outputs[j] * (1 - self.hidden_outputs[j]))

            # Update W2
            for j in range(self.hidden_size):
                for k in range(self.output_size):
                    self.W2[j][k] += self.lr * output_deltas[k] * self.hidden_outputs[j]

            # Update W1
            for i in range(self.input_size):
                for j in range(self.hidden_size):
                    self.W1[i][j] += self.lr * hidden_deltas[j] * inputs[i]