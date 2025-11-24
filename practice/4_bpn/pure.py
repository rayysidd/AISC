import math
import random

class PythonNN:
    def __init__(self, input_size, hidden_size, output_size, mode='binary'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = 0.1
        self.mode=mode
        # Init weights matrix manually
        self.W1=[]
        for i in range(input_size):
            row=[]
            for j in range(hidden_size):
                row.append(random.uniform(-0.5,0.5))
            self.W1.append(row)
        
        self.B1 = []
        for j in range(hidden_size):
            self.B1.append(random.uniform(-0.5, 0.5))

        self.W2 = []
        for i in range(hidden_size):
            row = []
            for j in range(output_size):
                row.append(random.uniform(-0.5, 0.5))
            self.W2.append(row) 

        self.B2 = []
        for j in range(hidden_size):
            self.B2.append(random.uniform(-0.5, 0.5))

    def activation(self,x):
        if self.mode == 'binary':
            return 1 / (1 + math.exp(-x))     # Sigmoid
        else:
            return (2 / (1 + math.exp(-x))) - 1

    def derivative(self,f_x):
        if self.mode == 'binary':
            return f_x * (1 - f_x)   # Sigmoid derivative
        else:
            return 0.5 * (1 + f_x) * (1 - f_x)  # Bipolar sigmoid derivative

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