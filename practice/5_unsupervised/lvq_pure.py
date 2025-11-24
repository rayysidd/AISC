import random

# -----------------------------
# 1. TRAINING DATA
# -----------------------------
X = [
    [80, 750, 30, 1, 25],  # A1
    [40, 600, 50, 1, 45],  # A2
    [90, 780, 20, 1, 20],  # A3
    [35, 550, 60, 0, 50],  # A4
    [70, 720, 35, 1, 30]   # A5
]

y = [
    [1, 0],  # Approved
    [0, 1],  # Rejected
    [1, 0],  # Approved
    [0, 1],  # Rejected
    [1, 0]   # Approved
]

ids = ["A1", "A2", "A3", "A4", "A5"]

# -----------------------------
# 2. NORMALIZATION (min-max)
# -----------------------------
def normalize(X):
    print("Normalizing features using min-max scaling...")
    num_features = len(X[0])
    max_vals = []

    # Step 1: find max of each feature
    for i in range(num_features):
        max_val = X[0][i]
        for row in X:
            if row[i] > max_val:
                max_val = row[i]
        max_vals.append(max_val)
    print("Max values per feature:", max_vals)

    # Step 2: divide each element by max
    X_norm = []
    for row in X:
        norm_row = []
        for i in range(num_features):
            norm_value = row[i] / max_vals[i]
            norm_row.append(norm_value)
        X_norm.append(norm_row)

    return X_norm, max_vals

X_scaled, X_max = normalize(X)

# -----------------------------
# 3. INITIAL PROTOTYPES (1 per class)
# -----------------------------
print("\nInitializing prototypes (1 per class)...")
W = []
W_labels = []

# Take first sample from each class as prototype
W.append(X_scaled[0])      # Prototype for Approved
W.append(X_scaled[1])      # Prototype for Rejected

W_labels.append([1, 0])    # Label Approved
W_labels.append([0, 1])    # Label Rejected

print("Initial prototypes:")
for i in range(len(W)):
    print("Prototype", i, ":", W[i], "Label:", W_labels[i])

learning_rate = 0.3

# -----------------------------
# 4. EUCLIDEAN DISTANCE
# -----------------------------
def euclidean(a, b):
    sum_sq = 0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_sq += diff * diff
    distance = sum_sq ** 0.5
    return distance

# -----------------------------
# 5. LVQ TRAINING
# -----------------------------
def lvq_train(X, y, W, W_labels, lr, epochs=10):
    print("\nStarting LVQ training...")
    for ep in range(epochs):
        print("\nEpoch", ep+1)
        for i in range(len(X)):
            sample = X[i]
            label = y[i]

            # Step 1: compute distances to all prototypes
            distances = []
            for w in W:
                dist = euclidean(sample, w)
                distances.append(dist)

            # Step 2: find winner prototype
            winner_idx = 0
            min_dist = distances[0]
            for j in range(1, len(distances)):
                if distances[j] < min_dist:
                    min_dist = distances[j]
                    winner_idx = j

            print("Sample", ids[i], "Label:", label)
            print("Distances to prototypes:", distances)
            print("Winner prototype index:", winner_idx, "Label:", W_labels[winner_idx])

            # Step 3: update winner
            if label == W_labels[winner_idx]:
                print("Correct class → moving prototype closer")
                for k in range(len(W[winner_idx])):
                    W[winner_idx][k] = W[winner_idx][k] + lr * (sample[k] - W[winner_idx][k])
            else:
                print("Incorrect class → moving prototype away")
                for k in range(len(W[winner_idx])):
                    W[winner_idx][k] = W[winner_idx][k] - lr * (sample[k] - W[winner_idx][k])

            print("Updated prototype:", W[winner_idx])

        # Optional learning rate decay
        lr *= 0.9
        print("Learning rate after decay:", lr)

    print("\nTraining finished.")
    return W

W_final = lvq_train(X_scaled, y, W.copy(), W_labels, learning_rate)

# -----------------------------
# 6. PREDICTION
# -----------------------------
def lvq_predict(sample, W, W_labels):
    distances = []
    for w in W:
        dist = euclidean(sample, w)
        distances.append(dist)

    winner_idx = 0
    min_dist = distances[0]
    for j in range(1, len(distances)):
        if distances[j] < min_dist:
            min_dist = distances[j]
            winner_idx = j

    return W_labels[winner_idx]

# -----------------------------
# 7. CLASSIFY TRAINING SAMPLES
# -----------------------------
print("\nFinal LVQ Predictions:")
for i in range(len(X_scaled)):
    sample = X_scaled[i]
    pred = lvq_predict(sample, W_final, W_labels)
    print(ids[i], "→ Predicted:", pred)
