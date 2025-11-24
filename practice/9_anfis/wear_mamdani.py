import math

# -------------------------
# DATASET
# -------------------------
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

# -------------------------
# TRAPEZOIDAL MEMBERSHIP FUNCTION
# -------------------------
def trapezoidal_mf(x, left, left_top, right_top, right):
    if x <= left or x >= right:
        return 0.0
    elif left < x < left_top:
        return (x - left) / (left_top - left)
    elif left_top <= x <= right_top:
        return 1.0
    elif right_top < x < right:
        return (right - x) / (right - right_top)
    else:
        return 0.0

# -------------------------
# MEMBERSHIP PARAMETERS
# -------------------------
cutting_speed_mfs = [(145, 155, 165, 175), (175, 180, 195, 205), (205, 215, 225, 235)]
feed_rate_mfs     = [(0.14, 0.16, 0.18, 0.20), (0.22, 0.24, 0.26, 0.28)]
vibration_mfs     = [(8, 10, 12, 14), (14, 16, 18, 20)]
temperature_mfs   = [(45, 48, 52, 55), (58, 60, 63, 65)]

# Output variable (tool wear) MFs for Mamdani
# Let's define 3 fuzzy sets: Low, Medium, High
tool_wear_mfs = [(0.0, 0.0, 0.15, 0.20), (0.15, 0.20, 0.25, 0.30), (0.25, 0.30, 0.35, 0.40)]

# -------------------------
# FORWARD PASS – COMPUTE RULE FIRING STRENGTHS
# -------------------------
def compute_rule_firing(sample):
    cs, fr, vb, tp = sample
    mu_cs = [trapezoidal_mf(cs, *mf) for mf in cutting_speed_mfs]
    mu_fr = [trapezoidal_mf(fr, *mf) for mf in feed_rate_mfs]
    mu_vb = [trapezoidal_mf(vb, *mf) for mf in vibration_mfs]
    mu_tp = [trapezoidal_mf(tp, *mf) for mf in temperature_mfs]

    # Build 24 rules as combinations of input MFs
    firing_strengths = []
    for i in range(len(mu_cs)):
        for j in range(len(mu_fr)):
            for k in range(len(mu_vb)):
                for l in range(len(mu_tp)):
                    firing_strengths.append(min(mu_cs[i], mu_fr[j], mu_vb[k], mu_tp[l]))  # Mamdani AND = min
    return firing_strengths

# -------------------------
# MAMDANI INFERENCE + DEFUZZIFICATION (CENTROID)
# -------------------------
def defuzzify(firing_strengths):
    # Weighted centroid approach
    numerator = 0.0
    denominator = 0.0
    num_rules = len(firing_strengths)
    for idx, w in enumerate(firing_strengths):
        # Map rule index to output MF (just simple mapping for example)
        output_mf_idx = idx % len(tool_wear_mfs)
        mf = tool_wear_mfs[output_mf_idx]
        peak = (mf[1] + mf[2]) / 2  # use center of plateau as representative
        numerator += w * peak
        denominator += w
    if denominator == 0:
        return 0.0
    return numerator / denominator

# -------------------------
# PREDICTION
# -------------------------
def predict_mamdani(sample):
    firing_strengths = compute_rule_firing(sample)
    return defuzzify(firing_strengths)

# -------------------------
# TRAIN + TEST
# -------------------------
predicted_outputs = [predict_mamdani(x) for x in input_samples]

# Compute RMSE
rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(predicted_outputs, target_outputs)) / len(target_outputs))

print("\n--- MAMDANI TOOL WEAR RESULTS (PURE PYTHON) ---\n")
print("Predictions:", [round(p, 3) for p in predicted_outputs])
print("Actual     :", target_outputs)
print("RMSE:", rmse)
