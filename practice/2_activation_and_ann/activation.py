import math 

def step(x):
    return 1 if x >= 0 else 0

def e(x):
    return math.exp(x)

def tanh(x):
    return (e(x)-e(-x))/(e(x)+e(-x))

def sigmoid(x):
    return 1/(1+e(-x))

def relu(x):
    return max(0,x)

def leakyRelu(x):
    return max(0.1*x,x)