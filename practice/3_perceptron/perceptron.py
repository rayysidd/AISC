import numpy as np


training_data = [
    ([1,0,0.7],1),
    ([1, 0, 0.9], 0),
    ([0, 1, 0.6], 1),
    ([0, 0, 0.95], 0),
]

def step(x):
    return 1 if x>=1 else 0

np.random.seed(42)
weights=np.random.randn(len(training_data[0][0]))
bias = np.random.randn()

lr=0.1
epochs=1000

for epoch in range(epochs):
    total_updates = 0
    for input,target in training_data:
        input=np.array(input)
        z=np.dot(input,weights)+bias
        pred=step(z)

        if pred!=target:
            if target==1:
                weights+= lr*target*input
                bias+=lr

            else:
                weights-= lr*target*input
                bias-=lr
            total_updates += 1
    if total_updates == 0:
            print(f"Training complete at epoch {epoch+1} as weight updates are 0")
            break
    print(f"Epoch {epoch+1}, Weight updates: {total_updates}")

print("\nFinal weights:", weights)
print("Final bias:", bias)


# --------------------------
# Testing
# --------------------------
print("\nFinal Weights:", weights)
print("Final Bias:", bias)
correct = 0

for inputs, target in training_data:
    inputs = np.array(inputs)
    z = np.dot(weights, inputs) + bias
    prediction = step(z)
    print(f"Input: {inputs}, Target: {target}, Predicted class: {'Apple' if prediction == 1 else 'Orange'}")
    if prediction==target:
        correct+=1

print("Total samples:", len(training_data))
print("Correct predictions:", correct)
print("Accuracy:", correct / len(training_data))