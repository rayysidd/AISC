import numpy as np

class MPNeuron:
    def __init__(self, threshold=None, weights=None):
        self.theta = threshold
        self.weights = weights

    def predict(self, x):
        # If weights are not provided, assume summation of binary inputs (Standard MP)
        if self.weights is None:
            activation = np.sum(x)
        else:
            # Weighted sum for more complex logic
            activation = np.dot(x, self.weights)
            
        return 1 if activation >= self.theta else 0

    def fit_threshold(self, X, Y):
        # Brute force search for best threshold (for B3, B4, A3)
        possible_thresholds = np.arange(0, len(X[0]) + 1)
        best_acc = -1
        best_theta = 0
        
        for theta in possible_thresholds:
            self.theta = theta
            preds = [self.predict(x) for x in X]
            acc = np.mean(np.array(preds) == np.array(Y))
            if acc > best_acc:
                best_acc = acc
                best_theta = theta
        
        self.theta = best_theta
        return best_acc, best_theta

# ==========================================
# APPLICATION: LOGIC CIRCUITS (B1, B2)
# ==========================================
def logic_gate(input1, input2, gate_type):
    # Standard MP params for gates
    if gate_type == "AND":
        neuron = MPNeuron(threshold=2)
        return neuron.predict([input1, input2])
    elif gate_type == "OR":
        neuron = MPNeuron(threshold=1)
        return neuron.predict([input1, input2])
    elif gate_type == "NOT":
        # Handled via negative weight logic or specialized threshold
        # For MP: inputs [x, 1], weights [-1, 1], threshold 1 (Inhibitory logic)
        val = input1 * -1 + 1 # Simplified simulation
        return 1 if val >= 1 else 0
    elif gate_type == "XOR":
        # XOR requires multi-layer: (A OR B) AND (NOT (A AND B))
        or_val = logic_gate(input1, input2, "OR")
        and_val = logic_gate(input1, input2, "AND")
        not_and = logic_gate(and_val, 0, "NOT") # 0 is dummy
        return logic_gate(or_val, not_and, "AND")

# --- Batch B1: Full Adder ---
def full_adder(a, b, cin):
    # Sum = A XOR B XOR Cin
    sum_ab = logic_gate(a, b, "XOR")
    final_sum = logic_gate(sum_ab, cin, "XOR")
    
    # Cout = (A AND B) OR (Cin AND (A XOR B))
    and_ab = logic_gate(a, b, "AND")
    xor_ab = logic_gate(a, b, "XOR")
    and_cin_xor = logic_gate(cin, xor_ab, "AND")
    cout = logic_gate(and_ab, and_cin_xor, "OR")
    
    return final_sum, cout

# --- Batch B2: 2-1 MUX ---
def mux_2to1(i0, i1, s):
    # Y = (NOT S AND I0) OR (S AND I1)
    not_s = logic_gate(s, 0, "NOT")
    term1 = logic_gate(not_s, i0, "AND")
    term2 = logic_gate(s, i1, "AND")
    y = logic_gate(term1, term2, "OR")
    return y

# ==========================================
# APPLICATION: CLASSIFIERS (B3, B4, A3)
# ==========================================
def run_classifier(batch_name, inputs, targets, feature_names):
    print(f"\n--- Processing {batch_name} ---")
    # For classification, we sometimes need 'inhibitory' weights 
    # (e.g., if a feature indicates the OPPOSITE class).
    # Here we assume inputs are processed such that 1 = supports class 1.
    
    model = MPNeuron()
    acc, theta = model.fit_threshold(inputs, targets)
    
    print(f"Best Threshold found: {theta}")
    print(f"Training Accuracy: {acc * 100}%")
    
    # Test
    print("Testing on Data:")
    for x, y in zip(inputs, targets):
        pred = model.predict(x)
        status = "✓" if pred == y else "✗"
        print(f"Input: {x} | Actual: {y} | Pred: {pred} {status}")

# --- Setup Data for Batches ---
if __name__ == "__main__":
    
    # B1 & B2 Example
    print(f"Full Adder (1, 0, 1) -> Sum, Cout: {full_adder(1, 0, 1)}")
    print(f"MUX (I0=1, I1=0, S=0) -> Output: {mux_2to1(1, 0, 0)}")

    # B3: Fruit (Apple=1, Orange=0)
    # Data: [Shape(Round=1), Texture(Smooth=1)]
    # Note: Orange is Round(1) but Rough(0). Apple is Round(1) & Smooth(1).
    b3_X = [[1, 1], [1, 0]] 
    b3_Y = [1, 0] 
    run_classifier("Batch B3 (Fruit)", b3_X, b3_Y, ["Shape", "Texture"])

    # B4: Toys (Bear=1, Rabbit=0) <-- Note logic inversion for easy MP sum
    # Features: [Ears_Long, Tail_Short, Body_Heavy, Climbs]
    # Rabbit (0): 1, 1, 0, 0
    # Bear (1):   0, 0, 1, 1
    # To use MP Sum for Bear(1), we need features that are present in Bear.
    # We can flip inputs for MP neuron or learn negative weights. 
    # Let's manually set weights: [-1, -1, 1, 1]
    b4_X = [[1,1,0,0], [0,0,1,1], [1,1,0,0]]
    b4_Y = [0, 1, 0] # Rabbit, Bear, Rabbit
    
    # Custom MP for B4 with weights
    print("\n--- Processing Batch B4 (Toys) ---")
    # Weights: Ears(-1), Tail(-1), Heavy(1), Climb(1) -> Threshold 1
    toy_neuron = MPNeuron(threshold=1, weights=[-1, -1, 1, 1])
    for x, y in zip(b4_X, b4_Y):
        print(f"Toy Input {x} -> Class {toy_neuron.predict(x)} (Expected {y})")