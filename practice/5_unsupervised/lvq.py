import numpy as np

class LVQ:
    def __init__(self, num_features, num_classes, learning_rate=0.5):
        self.num_features = num_features
        self.num_classes = num_classes
        self.lr = learning_rate

        self.weights = np.random.rand(num_classes, num_features)
        self.class_labels = np.arange(num_classes)

    def compute_dist(self, x, w):
        return np.sqrt(np.sum((x - w)**2))
    
    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            current_lr = self.lr * (1 - epoch / epochs)

            for i in range(len(X)):
                input_vector = X[i]
                target_class = y[i]

                # compute distances
                distances = np.linalg.norm(self.weights - input_vector, axis=1)
                winner_idx = np.argmin(distances)

                if winner_idx == target_class:
                    self.weights[winner_idx] += current_lr * (input_vector - self.weights[winner_idx])
                else:
                    self.weights[winner_idx] -= current_lr * (input_vector - self.weights[winner_idx])

    def predict(self, X):
        preds = []
        for vec in X:
            distances = np.linalg.norm(self.weights - vec, axis=1)
            preds.append(np.argmin(distances))
        return preds

# Data
X_A4 = np.array([
    [9.0, 95, 8, 6, 7],
    [8.0, 85, 4, 7, 9],
    [7.0, 70, 5, 5, 6],
    [6.5, 60, 3, 8, 8],
    [9.2, 90, 9, 4, 5],
    [8.5, 88, 5, 6, 7]
])
X_norm = X_A4 / X_A4.max(axis=0)
y_A4 = np.array([0, 2, 3, 2, 0, 1])

model = LVQ(num_features=5, num_classes=4, learning_rate=0.3)
model.train(X_norm, y_A4)

preds = model.predict(X_norm)
print("Predicted:", preds)
print("Actual:   ", list(y_A4))



# Features: [Income, Score, Amount, Employed, DebtRatio]
X_B3 = np.array([
    [80, 750, 30, 1, 25],
    [40, 600, 50, 1, 45],
    [90, 780, 20, 1, 20],
    [35, 550, 60, 0, 50],
    [70, 720, 35, 1, 30]
])
# Normalize!
X_norm = X_B3 / X_B3.max(axis=0)

# Targets: Approved=0, Rejected=1
y_B3 = np.array([0, 1, 0, 1, 0]) 

model = LVQ(num_features=5, num_classes=2, learning_rate=0.3)
model.train(X_norm, y_B3)

test_applicant = np.array([[40, 580, 55, 0, 48]]) # Likely Rejected
test_norm = test_applicant / X_B3.max(axis=0) # Use training max to scale
print("Predicted:", preds)
print("Actual:   ", list(y_B3))