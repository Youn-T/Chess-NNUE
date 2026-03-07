# IMPORTS
import numpy as np
import chess
import pandas as pd

print("Imports done")

# HYPERPARAMETERS
NEURONE_COUNT = 512
INPUT_COUNT = 768
BATCH_SIZE = 512
LEARNING_RATE = 0.5 # Augmenté pour compenser la taille du batch
EPOCHS = 10

# KEY FUNCTIONS
def ReLU(x):
    return np.maximum(0, x)
    
def CReLU(x):
    return np.clip(x, 0, 1)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def CReLU_derivative(x):
    return np.where((x > 0) & (x < 1), 1, 0)

def Sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def Sigmoid_derivative(x):
    return x * (1 - x)


def MSE(prediction, target):
    return np.square( prediction - target )


# FORWARD PASS
def forward_pass(input, W1, b1, W2, b2):
    
    # HIDDEN LAYER
    Z1 = np.dot(input,W1) + b1
    A1 = ReLU(Z1)

    # OUTPUT LAYER
    Z2 = np.dot(A1 , W2) + b2
    A2 = Sigmoid(Z2)
    return A1, Z1, A2

print("Functions defined, starting training...")
W1, b1, W2, b2 = init_weights(INPUT_COUNT, NEURONE_COUNT, 1)
for epoch in range(EPOCHS):
    indices = np.arange(len(BOARDS))
    np.random.shuffle(indices)
    
    BOARDS_SHUFFLED = BOARDS[indices]
    LABELS_SHUFFLED = LABELS[indices]
    
    for i in range(0, len(BOARDS), BATCH_SIZE):
        if (i+BATCH_SIZE) > len(BOARDS):
            break
        X_batch = BOARDS_SHUFFLED[i:i+BATCH_SIZE]
        y_batch = LABELS_SHUFFLED[i:i+BATCH_SIZE].reshape(-1, 1)
        
        A1, Z1, A2 = forward_pass(X_batch, W1, b1, W2, b2)
        
        rounded_predictions = np.round(A2.flatten())
        # accuracy = np.mean(rounded_predictions == y_batch.flatten())
        epsilon = 1e-15
        A2_clipped = np.clip(A2, epsilon, 1 - epsilon)

        # Calcul de la Binary Cross Entropy
        loss = -np.mean(y_batch * np.log(A2_clipped) + (1 - y_batch) * np.log(1 - A2_clipped))

        print(f"Epoch {epoch}, Batch {i//BATCH_SIZE}: Loss: {loss}")
        
        W1, b1, W2, b2 = backward_pass(X_batch, W1, b1, A1,Z1, W2, b2, A2, y_batch, LEARNING_RATE)
        
    print(f"End of Epoch {epoch}")
    
with open("model_weights.npz", "wb") as f:
    np.savez(f, W1=W1, b1=b1, W2=W2, b2=b2)
