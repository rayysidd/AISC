import numpy as np

# Training Data
# Inputs: [Shape, Texture, Weight]
# Shape: 1=Round, 0=Elliptical
# Texture: 1=Smooth, 0=Rough
# Weight: Normalized (0.5 ~120g, 0.9 ~200g etc.)
X = np.array([
    [1, 1, 0.7],   # Round, Smooth, ~150g -> Apple
    [1, 0, 0.9],   # Round, Rough, ~200g -> Orange
    [0, 1, 0.6],   # Elliptical, Smooth, ~120g -> Apple
    [1, 0, 0.95],  # Round, Rough, ~220g -> Orange
])

# Labels: 1 = Apple, 0 = Orange
y = np.array([1, 0, 1, 0])

#  Activation (Sigmoid for binary classification) 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize random weights 
np.random.seed(42)
weights = np.random.randn(3)   # w1, w2, w3
bias = np.random.randn()

#  Training Hyperparameters 
lr = 0.1
epochs = 1000

#  Training Loop (Gradient Descent) 
for epoch in range(epochs):
    # Linear combination
    z = np.dot(X, weights) + bias
    # Prediction
    y_hat = sigmoid(z)

    # Compute error (Binary cross-entropy derivative simplified)
    error = y - y_hat
    
    # Gradient update
    dw = np.dot(X.T, error * y_hat * (1 - y_hat)) / len(y)
    db = np.mean(error * y_hat * (1 - y_hat))
    
    weights += lr * dw
    bias += lr * db

#  Final Weights 
print("Trained Weights:", weights)
print("Trained Bias:", bias)

#  Testing the model 
test_sample = np.array([1, 1, 0.65])  # Round, Smooth, ~140g
pred = sigmoid(np.dot(test_sample, weights) + bias)
print("Prediction (Apple probability):", pred)
print("Class:", "Apple" if pred >= 0.5 else "Orange")


# Trained Weights: [-1.16815681  2.29702145 -0.87443093]
# Trained Bias: 0.5427819992686523
# Prediction (Apple probability): 0.7508716546883001
# Class: Apple