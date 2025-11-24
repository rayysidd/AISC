import math
import numpy as np

# --------------------------------------------------------
# DATASET
# --------------------------------------------------------
# Columns: [Cutting Speed, Feed Rate, Vibration, Temperature, Tool Wear]
data = [
    [150, 0.15, 10, 45, 0.12],
    [180, 0.20, 12, 55, 0.18],
    [200, 0.25, 15, 60, 0.25],
    [160, 0.18, 11, 48, 0.15],
    [170, 0.22, 14, 58, 0.20],
    [210, 0.28, 18, 65, 0.30],
    [155, 0.16, 10, 46, 0.13],
    [190, 0.24, 16, 62, 0.23],
    [175, 0.21, 13, 54, 0.18],
    [220, 0.30, 20, 70, 0.35]
]

input_samples = [row[:4] for row in data]
target_outputs = [row[4] for row in data]

# --------------------------------------------------------
# TRAPEZOIDAL MF FUNCTION
# --------------------------------------------------------
def trapezoidal_mf(x, left, left_top, right_top, right):
    if x <= left or x >= right:
        return 0.0
    elif left < x < left_top:
        return (x - left) / (left_top - left)
    elif left_top <= x <= right_top:
        return 1.0
    else:  # right_top < x < right
        return (right - x) / (right - right_top)

# --------------------------------------------------------
# MEMBERSHIP PARAMETERS (left, left_top, right_top, right)
# --------------------------------------------------------
cutting_speed_mfs = [
    (145, 155, 165, 175),  # Low
    (175, 180, 195, 205),  # Medium
    (205, 215, 225, 235)   # High
]

feed_rate_mfs = [
    (0.14, 0.16, 0.18, 0.20),  # Low
    (0.22, 0.24, 0.26, 0.28)   # High
]

vibration_mfs = [
    (8, 10, 12, 14),  # Low
    (14, 16, 18, 20)  # High
]

temperature_mfs = [
    (45, 48, 52, 55),  # Low
    (58, 60, 65, 68)   # High
]

# --------------------------------------------------------
# FORWARD PASS (Compute normalized firing strengths for 24 rules)
# --------------------------------------------------------
def compute_rule_weights(sample):
    cs, fr, vb, tp = sample

    mu_cs = [trapezoidal_mf(cs, *mf) for mf in cutting_speed_mfs]
    mu_fr = [trapezoidal_mf(fr, *mf) for mf in feed_rate_mfs]
    mu_vb = [trapezoidal_mf(vb, *mf) for mf in vibration_mfs]
    mu_tp = [trapezoidal_mf(tp, *mf) for mf in temperature_mfs]

    rule_weights = []
    # Total 3 * 2 * 2 * 2 = 24 rules
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    rule_weights.append(mu_cs[i] * mu_fr[j] * mu_vb[k] * mu_tp[l])

    total_weight = sum(rule_weights)
    if total_weight == 0:
        # Avoid zero division
        return [1.0 / 24] * 24
    return [w / total_weight for w in rule_weights]

# --------------------------------------------------------
# TRAIN CONSEQUENTS
# --------------------------------------------------------
def train_sugeno_consequents(X, y):
    num_rules = 24
    params = np.zeros((num_rules, 5))  # Each rule: [p1, p2, p3, p4, b]

    # Compute normalized weights for all samples
    all_rule_weights = [compute_rule_weights(sample) for sample in X]

    for r in range(num_rules):
        XtWX = np.zeros((5, 5))
        XtWy = np.zeros(5)

        for n, sample in enumerate(X):
            cs, fr, vb, tp = sample
            input_row = np.array([cs, fr, vb, tp, 1.0])
            w = all_rule_weights[n][r]

            XtWX += w * np.outer(input_row, input_row)
            XtWy += w * input_row * y[n]

        # Regularized pseudo-inverse to avoid singular matrices
        reg = 1e-6 * np.eye(5)
        params[r] = np.linalg.pinv(XtWX + reg) @ XtWy

    return params

# --------------------------------------------------------
# PREDICTION
# --------------------------------------------------------
def predict_sugeno(sample, params):
    rule_weights = compute_rule_weights(sample)
    outputs = []

    for r in range(24):
        p1, p2, p3, p4, b = params[r]
        cs, fr, vb, tp = sample
        outputs.append(p1*cs + p2*fr + p3*vb + p4*tp + b)

    # Weighted sum of rule outputs
    return sum(w * f for w, f in zip(rule_weights, outputs))

# --------------------------------------------------------
# TRAIN + TEST
# --------------------------------------------------------
sugeno_params = train_sugeno_consequents(input_samples, target_outputs)
predicted_outputs = [predict_sugeno(x, sugeno_params) for x in input_samples]

rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(predicted_outputs, target_outputs)) / len(target_outputs))

# --------------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------------
print("\n--- ANFIS TOOL WEAR RESULTS (Trapezoidal MF) ---\n")
print("Consequent Parameters for Each Rule:")
for i, rule_param in enumerate(sugeno_params, 1):
    print(f"Rule {i}: {rule_param.tolist()}")

print("\nPredicted Outputs:", [round(p, 3) for p in predicted_outputs])
print("Actual Outputs   :", target_outputs)
print("RMSE:", round(rmse, 5))
