import numpy as np
import pandas as pd
import chess
from scipy import sparse

# --- HYPERPARAMÈTRES ---
INPUT_COUNT = 768
LAYER_1_SIZE = 512
LAYER_2_SIZE = 256
LAYER_3_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001  # Standard pour Adam
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
ALPHA_LEAKY = 0.01  # Pente pour Leaky ReLU
LAMBDA_L2 = 0.0001

# --- FONCTIONS D'ACTIVATION ---
def Leaky_ReLU(x):
    return np.where(x > 0, x, x * ALPHA_LEAKY)

def Leaky_ReLU_derivative(x):
    return np.where(x > 0, 1, ALPHA_LEAKY)

def Sigmoid(x):
    # Clip pour éviter les overflows d'exponentielle
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def clip_gradients(grads, max_norm=1.0):
    for i in range(len(grads)):
        grads[i] = np.clip(grads[i], -max_norm, max_norm)
    return grads

# --- INITIALISATION DES POIDS (He Initialization) ---
W1 = np.random.randn(INPUT_COUNT, LAYER_1_SIZE) * np.sqrt(2.0 / 32.0)
b1 = np.zeros((1, LAYER_1_SIZE))
W2 = np.random.randn(LAYER_1_SIZE, LAYER_2_SIZE)* np.sqrt(2.0 / 32.0)
b2 = np.zeros((1, LAYER_2_SIZE))
W3 = np.random.randn(LAYER_2_SIZE, LAYER_3_SIZE) * np.sqrt(2.0 / LAYER_2_SIZE)
b3 = np.zeros((1, LAYER_3_SIZE))
W4 = np.random.randn(LAYER_3_SIZE, 1) * np.sqrt(2.0 / 32.0)
b4 = np.zeros((1, 1))

# Variables Adam
mW, vW = [np.zeros_like(W) for W in [W1, W2, W3, W4]], [np.zeros_like(W) for W in [W1, W2, W3, W4]]
mb, vb = [np.zeros_like(b) for b in [b1, b2, b3, b4]], [np.zeros_like(b) for b in [b1, b2, b3, b4]]

# --- PASSAGES ---
def forward_pass(X, weights, biases):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases
    
    Z1 = np.dot(X, W1) + b1
    A1 = Leaky_ReLU(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = Leaky_ReLU(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = Leaky_ReLU(Z3)
    
    Z4 = np.dot(A3, W4) + b4
    A4 = Sigmoid(Z4)
    
    return [A1, A2, A3, A4], [Z1, Z2, Z3, Z4]

def backward_pass(X, y, activations, zs, weights):
    A1, A2, A3, A4 = activations
    Z1, Z2, Z3, Z4 = zs
    W1, W2, W3, W4 = weights
    m = X.shape[0]

    dZ4 = A4 - y
    dW4 = np.dot(A3.T, dZ4) / m
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * Leaky_ReLU_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * Leaky_ReLU_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * Leaky_ReLU_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return [dW1, dW2, dW3, dW4], [db1, db2, db3, db4]

# --- CHARGEMENT DATASET (Simulé ici, utilise tes fichiers) ---
labels = np.load('labels.npz')["Y"]
moves = sparse.load_npz('moves.npz')

# --- BOUCLE D'ENTRAÎNEMENT ---
t = 0 # Compteur global Adam
weights = [W1, W2, W3, W4]
biases = [b1, b2, b3, b4]

for epoch in range(EPOCHS):
    indices = np.arange(moves.shape[0])
    np.random.shuffle(indices)
    total_loss = 0

    for i in range(0, len(indices), BATCH_SIZE):
        t += 1
        batch_idx = indices[i : i + BATCH_SIZE]
        X_batch = moves[batch_idx].toarray()
        y_batch = labels[batch_idx].reshape(-1, 1)

        # Forward
        activations, zs = forward_pass(X_batch, weights, biases)
        A4 = activations[-1]

        # Loss (BCE)
        loss = -np.mean(y_batch * np.log(A4 + 1e-12) + (1 - y_batch) * np.log(1 - A4 + 1e-12))
        total_loss += loss

        # Backward
        dWs, dbs = backward_pass(X_batch, y_batch, activations, zs, weights)
        dWs = clip_gradients(dWs)
        dbs = clip_gradients(dbs)
        # Adam Update
        for j in range(len(weights)):
            mW[j] = BETA1 * mW[j] + (1 - BETA1) * dWs[j]
            mb[j] = BETA1 * mb[j] + (1 - BETA1) * dbs[j]
            vW[j] = BETA2 * vW[j] + (1 - BETA2) * (dWs[j]**2)
            vb[j] = BETA2 * vb[j] + (1 - BETA2) * (dbs[j]**2)

            mw_hat = mW[j] / (1 - BETA1**t)
            mb_hat = mb[j] / (1 - BETA1**t)
            vw_hat = vW[j] / (1 - BETA2**t)
            vb_hat = vb[j] / (1 - BETA2**t)

            weights[j] -= LEARNING_RATE * (mw_hat / (np.sqrt(vw_hat) + EPSILON) + LAMBDA_L2 * weights[j])
            biases[j] -= LEARNING_RATE * (mb_hat / (np.sqrt(vb_hat) + EPSILON))

        if (i // BATCH_SIZE) % 200 == 0:
            # Moniteur de santé : % de valeurs activées positivement dans la couche 1
            alive_ratio = np.mean(zs[0] > 0)
            print(f"Epoch {epoch}, Batch {i//BATCH_SIZE} | Loss: {loss:.4f} | Neurones Actifs C1: {alive_ratio:.1%}")

    print(f"--- Fin Epoch {epoch}, Moyenne Loss: {total_loss / (len(indices)//BATCH_SIZE):.4f} ---")