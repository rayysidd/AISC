import numpy as np
# //created dataset in code
class BAM:
    """
    Bidirectional Associative Memory (BAM) Network
    This implementation uses bipolar vectors (1 and -1).
    """
    
    def __init__(self):
        self.W = None
        print("BAM network created.")

    def train(self, input_patterns, output_patterns):
        """Trains the BAM using a set of associated pairs."""
        
        if len(input_patterns) != len(output_patterns):
            raise ValueError("Input and output pattern lists must be the same length.")

        n = input_patterns[0].shape[0]
        m = output_patterns[0].shape[0]
        
        # Initialize the weight matrix W (size n x m)
        self.W = np.zeros((n, m))
        
        print(f"Training on {len(input_patterns)} pairs...")
        
        # Calculate the weight matrix using the Hebbian/Kosko rule
        for A, B in zip(input_patterns, output_patterns):
            A_col = A.reshape(-1, 1)  # Shape (n, 1)
            B_row = B.reshape(1, -1)  # Shape (1, m)
            self.W += A_col @ B_row
            
        print("Training complete.")

    def _activate(self, vector):
        """Bipolar threshold activation function."""
        return np.where(vector > 0, 1, -1)

    def recall_forward(self, A):
        """Recalls a pattern from Set A to Set B. (A -> B)"""
        if self.W is None:
            raise Exception("Model is not trained yet.")
        net_input = A @ self.W
        return self._activate(net_input)

    def recall_backward(self, B):
        """Recalls a pattern from Set B to Set A. (B -> A)"""
        if self.W is None:
            raise Exception("Model is not trained yet.")
        net_input = B @ self.W.T
        return self._activate(net_input)

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# A3 Example: "Music Recommendation"
# --- --- --- --- --- --- --- --- --- --- --- --- ---

print("### Starting A3: Music Recommendation Problem ###\n")

# 1. Define Mood & Genre patterns (as bipolar vectors)

# Set A: Moods
PAT_HAPPY = np.array([1, 1, 1, -1, -1, -1])
PAT_SAD = np.array([-1, -1, -1, 1, 1, 1])
PAT_ENERGETIC = np.array([1, -1, 1, -1, 1, -1])
PAT_ROMANTIC = np.array([1, 1, -1, -1, 1, 1]) # The "ambiguous" mood

# Set B: Genres
PAT_POP = np.array([1, -1, 1, -1, 1, 1])
PAT_BLUES = np.array([-1, 1, -1, 1, -1, -1])
PAT_EDM = np.array([1, 1, 1, 1, -1, -1])
PAT_CLASSICAL = np.array([-1, -1, 1, 1, 1, 1])
PAT_JAZZ = np.array([1, -1, -1, -1, 1, 1])

# 2. Construct the dataset (the pairs)
# We will create 5 associations
# The "Ambiguous" part: Romantic maps to both Classical and Jazz
input_patterns = [
    PAT_HAPPY,
    PAT_SAD,
    PAT_ENERGETIC,
    PAT_ROMANTIC,  # Pair 4
    PAT_ROMANTIC   # Pair 5
]

output_patterns = [
    PAT_POP,
    PAT_BLUES,
    PAT_EDM,
    PAT_CLASSICAL, # Pair 4: Romantic <-> Classical
    PAT_JAZZ       # Pair 5: Romantic <-> Jazz
]

# 3. Construct and test BAM
bam = BAM()
bam.train(input_patterns, output_patterns)

print("\n--- Testing Recall ---")

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# Test 1: Recall "Mood" -> "Song Genre"
# --- --- --- --- --- --- --- --- --- --- --- --- ---
print("\n### Test 1: Recall A -> B (Happy -> ?)")

recalled_pop = bam.recall_forward(PAT_HAPPY)
print(f"Input (Happy):   {PAT_HAPPY}")
print(f"Recalled:        {recalled_pop}")
print(f"Expected (Pop):  {PAT_POP}")
print(f"Recall successful: {np.array_equal(recalled_pop, PAT_POP)}")

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# Test 2: Recall "Song Genre" -> "Mood"
# --- --- --- --- --- --- --- --- --- --- --- --- ---
print("\n### Test 2: Recall B -> A (EDM -> ?)")

recalled_energetic = bam.recall_backward(PAT_EDM)
print(f"Input (EDM):     {PAT_EDM}")
print(f"Recalled:        {recalled_energetic}")
print(f"Expected (Energetic): {PAT_ENERGETIC}")
print(f"Recall successful: {np.array_equal(recalled_energetic, PAT_ENERGETIC)}")

# --- --- --- --- --- --- --- --- --- --- --- --- ---
# Test 3: Test "Ambiguous Mood" (Romantic -> ?)
# --- --- --- --- --- --- --- --- --- --- --- --- ---
print("\n### Test 3: Ambiguous Mood (Romantic -> ?)")
print("This tests the one-to-many mapping.")

recalled_ambiguous = bam.recall_forward(PAT_ROMANTIC)
print(f"Input (Romantic): {PAT_ROMANTIC}")
print(f"Recalled:         {recalled_ambiguous}")

print("\n--- Analysis of Ambiguity ---")
print(f"Expected (Classical): {PAT_CLASSICAL}")
print(f"Expected (Jazz):      {PAT_JAZZ}")

is_classical = np.array_equal(recalled_ambiguous, PAT_CLASSICAL)
is_jazz = np.array_equal(recalled_ambiguous, PAT_JAZZ)

print(f"Is recall Classical? {is_classical}")
print(f"Is recall Jazz? {is_jazz}")

if not is_classical and not is_jazz:
    print("\nSUCCESS: The recall is 'confused'!")
    print("The recalled pattern is not Classical and not Jazz.")
    print("This is because 'Romantic' is associated with *both* genres.")
    print("The network returns a muddled superposition of the two patterns.")