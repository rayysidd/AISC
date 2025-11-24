import numpy as np
import matplotlib.pyplot as plt

class FuzzyLogicSystem:
    def __init__(self, output_range):
        self.rules = []
        self.output_range = output_range # e.g., np.linspace(0, 100, 100)
        self.aggregated_fuzzy_output = np.zeros_like(output_range)

    # --- Membership Functions ---
    def trimf(self, x, params):
        """Triangular Membership Function [a, b, c]"""
        a, b, c = params
        if a == b: return np.maximum(0, np.minimum((c - x) / (c - b), 1)) # Edge case
        if b == c: return np.maximum(0, np.minimum((x - a) / (b - a), 1))
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

    def trapmf(self, x, params):
        """Trapezoidal Membership Function [a, b, c, d]"""
        a, b, c, d = params
        # Handle edges (left/right shoulders)
        left = (x - a) / (b - a) if b > a else np.ones_like(x)
        right = (d - x) / (d - c) if d > c else np.ones_like(x)
        return np.maximum(0, np.minimum(np.minimum(left, right), 1))

    # --- Inference Engine ---
    def add_rule(self, input_memberships, output_mf_params, mf_type='trimf'):
        """
        input_memberships: list of degree of membership [0.2, 0.5, ...]
        output_mf_params: parameters for the output set
        """
        # 1. Apply 'AND' operator (Min) to inputs (Mamdani Inference)
        firing_strength = np.min(input_memberships)
        
        # 2. Implication (Clip the output MF by firing strength)
        if mf_type == 'trimf':
            raw_output = self.trimf(self.output_range, output_mf_params)
        else:
            raw_output = self.trapmf(self.output_range, output_mf_params)
            
        clipped_output = np.minimum(firing_strength, raw_output)
        
        # 3. Aggregation (OR operator -> Max)
        self.aggregated_fuzzy_output = np.maximum(self.aggregated_fuzzy_output, clipped_output)

    def reset_aggregation(self):
        self.aggregated_fuzzy_output = np.zeros_like(self.output_range)

    # --- Defuzzification ---
    def defuzzify(self, method='centroid'):
        if method == 'centroid':
            numerator = np.sum(self.output_range * self.aggregated_fuzzy_output)
            denominator = np.sum(self.aggregated_fuzzy_output)
            if denominator == 0: return 0 # Handle no activation
            return numerator / denominator
        
        elif method == 'mom': # Mean of Maxima (For Batch B4)
            max_val = np.max(self.aggregated_fuzzy_output)
            if max_val == 0: return 0
            indices = np.where(self.aggregated_fuzzy_output == max_val)[0]
            return np.mean(self.output_range[indices])

    def plot_aggregation(self):
        plt.fill_between(self.output_range, self.aggregated_fuzzy_output, alpha=0.5)
        plt.title("Aggregated Fuzzy Output")
        plt.show()

# --- SETUP FOR BATCH A1 / B1 ---
# Select your batch settings below
BATCH = "A1" # Change to "B1" for Pendulum

if BATCH == "A1":
    # Ball and Beam Parameters
    labels = ['NB', 'NS', 'Z', 'PS', 'PB']
    # Trapezoidal Params: [a, b, c, d]
    # Error Input Range: -5 to 5
    in_mfs = {
        'NB': [-10, -10, -3, -1], 'NS': [-3, -1, -0.5, 0], 
        'Z':  [-0.5, -0.1, 0.1, 0.5], 
        'PS': [0, 0.5, 1, 3], 'PB': [1, 3, 10, 10]
    }
    # Output Range (Beam Angle): -45 to 45 deg
    x_out = np.linspace(-45, 45, 500)
    out_mfs = {
        'NB': [-60, -60, -30, -15], 'NS': [-30, -15, -5, 0],
        'Z':  [-5, -1, 1, 5],
        'PS': [0, 5, 15, 30], 'PB': [15, 30, 60, 60]
    }
    
elif BATCH == "B1":
    # Inverted Pendulum Parameters
    labels = ['NB', 'NS', 'Z', 'PS', 'PB']
    # Error Input Range (Theta): -90 to 90
    in_mfs = {
        'NB': [-180, -180, -45, -15], 'NS': [-45, -15, -5, 0],
        'Z':  [-5, -1, 1, 5],
        'PS': [0, 5, 15, 45], 'PB': [15, 45, 180, 180]
    }
    # Output Range (Force): -50 to 50 Newtons
    x_out = np.linspace(-50, 50, 500)
    out_mfs = {
        'NB': [-100, -100, -40, -20], 'NS': [-40, -20, -5, 0],
        'Z':  [-5, -1, 1, 5],
        'PS': [0, 5, 20, 40], 'PB': [20, 40, 100, 100]
    }

flc = FuzzyLogicSystem(x_out)

def get_control_action(error, d_error):
    flc.reset_aggregation()
    
    # Fuzzify Inputs (Calculate membership degree)
    # Note: Using trapmf for both as requested
    e_mems = {l: flc.trapmf(error, in_mfs[l]) for l in labels}
    de_mems = {l: flc.trapmf(d_error, in_mfs[l]) for l in labels}

    # --- RULE BASE (Standard Diagonal Matrix) ---
    # Rule syntax: IF Error is NB AND dError is NB THEN Output is PB (Counteract)
    # Simple Logic: If error is Negative (left), push Positive (right)
    
    # 1. Extreme Correction
    flc.add_rule([e_mems['NB'], de_mems['NB']], out_mfs['PB'], 'trapmf')
    flc.add_rule([e_mems['PB'], de_mems['PB']], out_mfs['NB'], 'trapmf')
    
    # 2. Standard Correction
    flc.add_rule([e_mems['NS'], de_mems['Z']], out_mfs['PS'], 'trapmf')
    flc.add_rule([e_mems['PS'], de_mems['Z']], out_mfs['NS'], 'trapmf')
    
    # 3. Stability
    flc.add_rule([e_mems['Z'], de_mems['Z']], out_mfs['Z'], 'trapmf')
    
    # (Add more rules for full coverage NB/NS/Z/PS/PB cross products)
    
    return flc.defuzzify('centroid')

# --- PHYSICS SIMULATION LOOP ---
time_points = np.linspace(0, 10, 100)
position = []
current_pos = 2.0 # Initial Displacement (e.g., 2cm or 2 deg)
velocity = 0.0

print(f"Starting Simulation for {BATCH}...")
print(f"Initial Pos: {current_pos}")

for t in time_points:
    # 1. Disturbance Test (Tap at t=5s)
    if 4.9 < t < 5.1:
        velocity += 0.5 # Impulse
        
    # 2. Calculate Inputs
    ref = 0
    error = ref - current_pos
    d_error = -velocity
    
    # 3. Get Fuzzy Control Output
    control_action = get_control_action(error, d_error)
    
    # 4. Apply Physics (Simplified approximation)
    # A1 BallBeam: Angle -> Acceleration
    # B1 Pendulum: Force -> Angular Acceleration
    acceleration = control_action * 0.5 - 0.1 * velocity # 0.1 is friction
    
    # Euler Integration
    velocity += acceleration * (time_points[1] - time_points[0])
    current_pos += velocity * (time_points[1] - time_points[0])
    
    position.append(current_pos)

# --- PLOTTING ---
plt.plot(time_points, position)
plt.title(f"{BATCH} Fuzzy Control Response")
plt.xlabel("Time (s)")
plt.ylabel("Position/Angle")
plt.grid(True)
plt.axhline(0, color='r', linestyle='--') # Target
plt.show()

# --- COMPARISON METRICS CALCULATION ---
position = np.array(position)
settling_time = time_points[np.where(np.abs(position) < 0.1)[0][0]] if np.any(np.abs(position) < 0.1) else 10
overshoot = np.max(position) - 0 # Assuming start at positive
print(f"Settling Time: {settling_time:.2f}s")
print(f"Max Overshoot: {overshoot:.2f}")


# --- SETUP FOR DIAGNOSTIC BATCHES ---
BATCH = "B4" # Change to A2, A4, B2, B3, B4

# 1. Define Output Range (Score 0-100 or Risk Index)
x_out = np.linspace(0, 100, 1000)
flc = FuzzyLogicSystem(x_out)

# 2. Configure MFs based on Batch
if BATCH == "B4": # Diabetic Foot
    # Inputs: Redness, Lesion Area (Normalized 0-10)
    # MF Shape: Triangular
    labels = ['Low', 'Med', 'High']
    in_mfs = {
        'Low': [0, 0, 5], 'Med': [2, 5, 8], 'High': [5, 10, 10]
    }
    # Output: Risk Index (Normal, At-Risk, Severe)
    out_labels = ['Normal', 'Risk', 'Severe']
    out_mfs = {
        'Normal': [0, 0, 40], 'Risk': [30, 50, 70], 'Severe': [60, 100, 100]
    }
    DEFUZZ_METHOD = 'mom' # Mean of Maxima requested
    MF_TYPE = 'trimf'

elif BATCH == "A2": # Battery SOH
    # Inputs: Resistance, Fade (Normalized 0-10)
    # MF Shape: Trapezoidal
    labels = ['Low', 'Med', 'High']
    in_mfs = {
        'Low': [0, 0, 2, 4], 'Med': [2, 4, 6, 8], 'High': [6, 8, 10, 10]
    }
    out_labels = ['Bad', 'Fair', 'Good']
    out_mfs = {
        'Bad': [0, 0, 30, 40], 'Fair': [30, 40, 60, 70], 'Good': [60, 70, 100, 100]
    }
    DEFUZZ_METHOD = 'centroid'
    MF_TYPE = 'trapmf'

# (Add elif blocks for A4, B2, B3 with similar structures)

def evaluate_sample(inputs):
    # inputs = list of values [val1, val2...]
    flc.reset_aggregation()
    
    # 1. Fuzzify Inputs
    # For simplicity, assuming all inputs share same MF ranges (0-10 scale)
    # In real code, define specific MFs per input variable
    mems = [] # Store memberships for each input
    for val in inputs:
        val_mems = {}
        if MF_TYPE == 'trimf':
            for l in labels: val_mems[l] = flc.trimf(val, in_mfs[l])
        else:
            for l in labels: val_mems[l] = flc.trapmf(val, in_mfs[l])
        mems.append(val_mems)
        
    # 2. Rule Evaluation
    # Example Rule: IF Input1 is High AND Input2 is High THEN Output is Severe/Bad
    
    # Rule 1: High Inputs -> High Output (Risk/Damage)
    # Taking Min of "High" membership of Input 1 and "High" membership of Input 2
    rule1_strength = np.min([m[labels[-1]] for m in mems]) 
    flc.add_rule([rule1_strength], out_mfs[out_labels[-1]], MF_TYPE)
    
    # Rule 2: Low Inputs -> Low Output (Normal/Good)
    rule2_strength = np.min([m[labels[0]] for m in mems])
    flc.add_rule([rule2_strength], out_mfs[out_labels[0]], MF_TYPE)
    
    # Rule 3: Mixed -> Med Output
    # Ideally write specific logic, here generalizing
    rule3_strength = 1 - max(rule1_strength, rule2_strength)
    if len(out_labels) > 2:
        flc.add_rule([rule3_strength], out_mfs[out_labels[1]], MF_TYPE)

    return flc.defuzzify(DEFUZZ_METHOD)

# --- SIMULATE DATASET ---
# Generating synthetic data [Input1, Input2, ...]
num_samples = 20
synthetic_data = np.random.uniform(0, 10, (num_samples, 2)) # 2 inputs
results = []

print(f"Evaluating {BATCH} System...")
print(f"Inputs (0-10) | Output Score (0-100) | Interpretation")

for sample in synthetic_data:
    score = evaluate_sample(sample)
    results.append(score)
    
    # Interpret
    interp = "Unknown"
    if score < 33: interp = out_labels[0]
    elif score < 66: interp = out_labels[1] if len(out_labels)>1 else out_labels[-1]
    else: interp = out_labels[-1]
    
    print(f"{sample} | {score:.2f} | {interp}")

# --- COMPARISON / SENSITIVITY ---
# Add noise to data as requested in Batches A2/B4
noisy_data = synthetic_data + np.random.normal(0, 0.5, synthetic_data.shape) # Gaussian Noise
print("\n--- Sensitivity Analysis (Noisy Data) ---")
for i in range(3): # Show first 3
    orig = results[i]
    new = evaluate_sample(noisy_data[i])
    print(f"Original: {orig:.2f} -> Noisy: {new:.2f} (Diff: {abs(orig-new):.2f})")