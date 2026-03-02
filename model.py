import cupy as cp
import numpy as np
import pandas as pd
import chess
from scipy import sparse
import cupyx.scipy.sparse as cp_sparse

# --- HYPERPARAMÈTRES ---
INPUT_COUNT = 40960
LAYER_1_SIZE = 512
LAYER_2_SIZE = 32
LAYER_3_SIZE = 32
BATCH_SIZE = 8192 # Précedemment à 128
EPOCHS = 30
LEARNING_RATE = 0.001  # Standard pour Adam
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
ALPHA_LEAKY = 0.01  # Pente pour Leaky ReLU
LAMBDA_L2 = 0.00001 # Précédemment 0.0001, à ajuster selon les résultats

# --- FONCTIONS D'ACTIVATION ---

def Leaky_Clipped_ReLU(x):
    # On applique la pente leaky en bas et on coupe à 1.0 en haut
    return cp.where(x > 1.0, 1.0, cp.where(x > 0, x, x * ALPHA_LEAKY))

def Leaky_Clipped_ReLU_derivative(x):
    # 1.0 entre 0 et 1, ALPHA en dessous de 0, et 0 au-dessus de 1
    grad = cp.where((x >= 0) & (x <= 1.0), 1.0, 0.0)
    grad = cp.where(x < 0, ALPHA_LEAKY, grad)
    return grad

def Leaky_ReLU(x):
    return cp.where(x > 0, x, x * ALPHA_LEAKY)

def Leaky_ReLU_derivative(x):
    return cp.where(x > 0, 1, ALPHA_LEAKY)

def Sigmoid(x):
    # Clip pour éviter les overflows d'exponentielle
    x = cp.clip(x, -500, 500)
    return 1 / (1 + cp.exp(-x))

def clip_gradients(grads, max_norm=1.0):
    for i in range(len(grads)):
        grads[i] = cp.clip(grads[i], -max_norm, max_norm)
    return grads

# --- INITIALISATION DES POIDS (He Initialization) ---
W1 = cp.random.randn(INPUT_COUNT, LAYER_1_SIZE) * cp.sqrt(2.0 / 32.0)
b1 = cp.zeros((1, LAYER_1_SIZE))
W2 = cp.random.randn(LAYER_1_SIZE, LAYER_2_SIZE) * cp.sqrt(2.0 / LAYER_1_SIZE)
b2 = cp.zeros((1, LAYER_2_SIZE))
W3 = cp.random.randn(LAYER_2_SIZE, LAYER_3_SIZE) * cp.sqrt(2.0 / LAYER_2_SIZE)
b3 = cp.zeros((1, LAYER_3_SIZE))
W4 = cp.random.randn(LAYER_3_SIZE, 1) * cp.sqrt(2.0 / LAYER_3_SIZE)
b4 = cp.zeros((1, 1))

# Variables Adam
mW, vW = [cp.zeros_like(W) for W in [W1, W2, W3, W4]], [cp.zeros_like(W) for W in [W1, W2, W3, W4]]
mb, vb = [cp.zeros_like(b) for b in [b1, b2, b3, b4]], [cp.zeros_like(b) for b in [b1, b2, b3, b4]]

# --- PASSAGES ---
def forward_pass(X, weights, biases):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases
    
    Z1 = X.dot(W1) + b1
    A1 = Leaky_Clipped_ReLU(Z1)
    
    Z2 = cp.dot(A1, W2) + b2
    A2 = Leaky_ReLU(Z2)
    
    Z3 = cp.dot(A2, W3) + b3
    A3 = Leaky_ReLU(Z3)
    
    Z4 = cp.dot(A3, W4) + b4
    A4 = Sigmoid(Z4)
    
    return [A1, A2, A3, A4], [Z1, Z2, Z3, Z4]

def backward_pass(X, y, activations, zs, weights):
    A1, A2, A3, A4 = activations
    Z1, Z2, Z3, Z4 = zs
    W1, W2, W3, W4 = weights
    m = X.shape[0]

    dZ4 = A4 - y
    dW4 = cp.dot(A3.T, dZ4) / m
    db4 = cp.sum(dZ4, axis=0, keepdims=True) / m

    dA3 = cp.dot(dZ4, W4.T)
    dZ3 = dA3 * Leaky_ReLU_derivative(Z3)
    dW3 = cp.dot(A2.T, dZ3) / m
    db3 = cp.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = cp.dot(dZ3, W3.T)
    dZ2 = dA2 * Leaky_ReLU_derivative(Z2)
    dW2 = cp.dot(A1.T, dZ2) / m
    db2 = cp.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = cp.dot(dZ2, W2.T)
    dZ1 = dA1 * Leaky_Clipped_ReLU_derivative(Z1)
    dW1 = X.T.dot(dZ1) / m
    db1 = cp.sum(dZ1, axis=0, keepdims=True) / m

    return [dW1, dW2, dW3, dW4], [db1, db2, db3, db4]

# --- CHARGEMENT DATASET (Simulé ici, utilise tes fichiers) ---
labels = cp.load('labels_halfKP_V1.npz')["Y"]
matrix_cpu = sparse.load_npz('moves_halfKP_V1.npz')
moves = cp_sparse.csr_matrix(matrix_cpu)  # Transfert vers GPU
# --- BOUCLE D'ENTRAÎNEMENT ---
t = 0 # Compteur global Adam
weights = [W1, W2, W3, W4]
biases = [b1, b2, b3, b4]

for epoch in range(EPOCHS):
    indices = cp.arange(moves.shape[0])
    cp.random.shuffle(indices)
    total_loss = 0

    if epoch < 15:
        current_lr = LEARNING_RATE
    elif epoch < 25:
        current_lr = LEARNING_RATE * 0.1
    else:
        current_lr = LEARNING_RATE * 0.01

    for i in range(0, len(indices), BATCH_SIZE):
        t += 1
        # current_lr = LEARNING_RATE * (0.1 ** (t / 150000))
        batch_idx = indices[i : i + BATCH_SIZE]
        X_batch = moves[batch_idx]
        y_batch = labels[batch_idx].reshape(-1, 1)

        # Forward
        activations, zs = forward_pass(X_batch, weights, biases)
        A4 = activations[-1]

        # Loss (BCE)
        loss = -cp.mean(y_batch * cp.log(A4 + 1e-12) + (1 - y_batch) * cp.log(1 - A4 + 1e-12))
        total_loss += loss

        # Backward
        dWs, dbs = backward_pass(X_batch, y_batch, activations, zs, weights)
        dWs = clip_gradients(dWs)
        dbs = clip_gradients(dbs)
        
        # Adam Update
        # ==========================================================
        # 1. ADAM SPARSE POUR LA COUCHE 1 (L'Accumulateur HalfKP)
        # ==========================================================
        # On extrait les indices des colonnes actives dans ce batch spécifique
        active_cols = cp.unique(X_batch.indices) 
        
        # On ne met à jour que les moments et les poids des pièces VUES
        mW[0][active_cols] = BETA1 * mW[0][active_cols] + (1 - BETA1) * dWs[0][active_cols]
        vW[0][active_cols] = BETA2 * vW[0][active_cols] + (1 - BETA2) * (dWs[0][active_cols]**2)
        mb[0] = BETA1 * mb[0] + (1 - BETA1) * dbs[0]
        vb[0] = BETA2 * vb[0] + (1 - BETA2) * (dbs[0]**2)

        mw_hat_0 = mW[0][active_cols] / (1 - BETA1**t)
        vw_hat_0 = vW[0][active_cols] / (1 - BETA2**t)
        mb_hat_0 = mb[0] / (1 - BETA1**t)
        vb_hat_0 = vb[0] / (1 - BETA2**t)

        # Le L2 n'est appliqué qu'aux poids actifs, stoppant "l'amnésie" du réseau
        weights[0][active_cols] -= current_lr * (mw_hat_0 / (cp.sqrt(vw_hat_0) + EPSILON) + LAMBDA_L2 * weights[0][active_cols])
        biases[0] -= current_lr * (mb_hat_0 / (cp.sqrt(vb_hat_0) + EPSILON))

        # ==========================================================
        # 2. ADAM DENSE STANDARD POUR LES COUCHES 2, 3 ET 4
        # ==========================================================
        for j in range(1, len(weights)):
            mW[j] = BETA1 * mW[j] + (1 - BETA1) * dWs[j]
            mb[j] = BETA1 * mb[j] + (1 - BETA1) * dbs[j]
            vW[j] = BETA2 * vW[j] + (1 - BETA2) * (dWs[j]**2)
            vb[j] = BETA2 * vb[j] + (1 - BETA2) * (dbs[j]**2)

            mw_hat = mW[j] / (1 - BETA1**t)
            mb_hat = mb[j] / (1 - BETA1**t)
            vw_hat = vW[j] / (1 - BETA2**t)
            vb_hat = vb[j] / (1 - BETA2**t)

            weights[j] -= current_lr * (mw_hat / (cp.sqrt(vw_hat) + EPSILON) + LAMBDA_L2 * weights[j])
            biases[j] -= current_lr * (mb_hat / (cp.sqrt(vb_hat) + EPSILON))

        if (i // BATCH_SIZE) % 200 == 0:
            # Moniteur de santé : % de valeurs activées positivement dans la couche 1
            alive_mask = (zs[0] > 0) & (zs[0] <= 1.0)  # Neurones actifs dans la plage de Leaky_Clipped_ReLU
            alive_ratio = cp.mean(alive_mask)
            print(f"Epoch {epoch}, Batch {i//BATCH_SIZE} | Loss: {loss:.4f} | Neurones Actifs C1: {alive_ratio:.1%}")

    print(f"--- Fin Epoch {epoch}, Moyenne Loss: {total_loss / (len(indices)//BATCH_SIZE):.4f} ---")

cp.savez('model_halfKP_V1.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
print("Modèle sauvegardé sous 'model_halfKP_V1.npz'")