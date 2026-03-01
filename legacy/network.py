# IMPORTS
import numpy as np
import chess
import pandas as pd

print("Imports done")

# KEY FUNCTIONS
def ReLU(x):
    return np.maximum(0, x)
    
def Sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# FORWARD PASS
def forward_pass(input, W1, b1, W2, b2):
    
    # HIDDEN LAYER
    Z1 = np.dot(input,W1) + b1
    A1 = ReLU(Z1)

    # OUTPUT LAYER
    Z2 = np.dot(A1 , W2) + b2
    A2 = Sigmoid(Z2)
    return A1, Z1, A2


X = np.random.rand(1, 768)  # Exemple d'entrée aléatoire

with np.load("model_weights.npz") as data:
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    
    A1, Z1, A2 = forward_pass(X, W1, b1, W2, b2)
    print(A2)

def predict(input, W1, b1, W2, b2):
    _, _, A2 = forward_pass(input, W1, b1, W2, b2)
    return A2