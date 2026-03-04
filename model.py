import cupy as cp
import numpy as np
import pandas as pd
import chess
from scipy import sparse
import cupyx.scipy.sparse as cp_sparse

import time

# --- HYPERPARAMÈTRES ---
INPUT_COUNT = 40960
LAYER_1_SIZE = 256  # Par perspective (256 us + 256 them = 512 concaténés, comme Stockfish NNUE)
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
        cp.clip(grads[i], -max_norm, max_norm, out=grads[i])
    return grads

@cp.fuse()
def bce_loss_fused(y, a):
    return -(y * cp.log(a + 1e-12) + (1 - y) * cp.log(1 - a + 1e-12))

# --- INITIALISATION DES POIDS (He Initialization) ---
W1 = cp.random.randn(INPUT_COUNT, LAYER_1_SIZE) * cp.sqrt(2.0 / 32.0)
b1 = cp.zeros((1, LAYER_1_SIZE))
W2 = cp.random.randn(LAYER_1_SIZE * 2, LAYER_2_SIZE) * cp.sqrt(2.0 / LAYER_1_SIZE)
b2 = cp.zeros((1, LAYER_2_SIZE))
W3 = cp.random.randn(LAYER_2_SIZE, LAYER_3_SIZE) * cp.sqrt(2.0 / LAYER_2_SIZE)
b3 = cp.zeros((1, LAYER_3_SIZE))
W4 = cp.random.randn(LAYER_3_SIZE, 1) * cp.sqrt(2.0 / LAYER_3_SIZE)
b4 = cp.zeros((1, 1))

# Variables Adam
mW, vW = [cp.zeros_like(W) for W in [W1, W2, W3, W4]], [cp.zeros_like(W) for W in [W1, W2, W3, W4]]
mb, vb = [cp.zeros_like(b) for b in [b1, b2, b3, b4]], [cp.zeros_like(b) for b in [b1, b2, b3, b4]]

# --- PASSAGES ---
def forward_pass(X_us, X_them, weights, biases):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases
    
    # 2 sparse matmuls séparées (évite vstack + overhead)
    Z1_us = X_us.dot(W1) + b1
    A1_us = Leaky_Clipped_ReLU(Z1_us)
    
    Z1_them = X_them.dot(W1) + b1
    A1_them = Leaky_Clipped_ReLU(Z1_them)
    
    A1 = cp.concatenate([A1_us, A1_them], axis=1)
    
    Z2 = cp.dot(A1, W2) + b2
    A2 = Leaky_ReLU(Z2)
    
    Z3 = cp.dot(A2, W3) + b3
    A3 = Leaky_ReLU(Z3)
    
    Z4 = cp.dot(A3, W4) + b4
    A4 = Sigmoid(Z4)
    
    return [A1, A2, A3, A4], [Z1_us, Z1_them, Z2, Z3, Z4]

def backward_pass(X_us, X_them, y, activations, zs, weights):
    A1, A2, A3, A4 = activations
    Z1_us, Z1_them = zs[0], zs[1]
    Z2, Z3, Z4 = zs[2], zs[3], zs[4]
    W1, W2, W3, W4 = weights
    m = y.shape[0]

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
    
    dA1_us = dA1[:, :LAYER_1_SIZE]
    dZ1_us = dA1_us * Leaky_Clipped_ReLU_derivative(Z1_us)
    
    dA1_them = dA1[:, LAYER_1_SIZE:]
    dZ1_them = dA1_them * Leaky_Clipped_ReLU_derivative(Z1_them)
    
    # 2 sparse .T.dot() séparées (CSC implicite, PAS de conversion CSR)
    dW1 = (X_us.T.dot(dZ1_us) + X_them.T.dot(dZ1_them)) / m
    db1 = (cp.sum(dZ1_us, axis=0, keepdims=True) + cp.sum(dZ1_them, axis=0, keepdims=True)) / m

    return [dW1, dW2, dW3, dW4], [db1, db2, db3, db4]

# --- CHARGEMENT DATASET ---
# Labels restent en RAM CPU (52 MB)
labels_cpu = np.load('labels_halfKP_V4.npz')["Y"]

# Matrices sparse en float16 sur CPU (~986 MB chacune au lieu de 1972 MB)
matrix_us_cpu = sparse.load_npz('moves_halfKP_V4_us.npz').astype(np.bool)
matrix_them_cpu = sparse.load_npz('moves_halfKP_V4_them.npz').astype(np.bool)

num_samples = matrix_us_cpu.shape[0]
print(f"Dataset: {num_samples:,} samples, sparse float16 sur CPU")
# --- BOUCLE D'ENTRAÎNEMENT ---
t = 0 # Compteur global Adam
weights = [W1, W2, W3, W4]
biases = [b1, b2, b3, b4]

t0 = time.perf_counter()
tpast = t0

for epoch in range(EPOCHS):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    total_loss = 0

    if epoch < 15:
        current_lr = LEARNING_RATE
    elif epoch < 25:
        current_lr = LEARNING_RATE * 0.1
    else:
        current_lr = LEARNING_RATE * 0.01

    for i in range(0, num_samples, BATCH_SIZE):
        t += 1
        batch_idx = indices[i : i + BATCH_SIZE]
        
        # Extraction CPU (float16 CSR) puis transfert GPU (converti en float32 sur GPU)
        X_batch_us = cp_sparse.csr_matrix(matrix_us_cpu[batch_idx].astype(np.float32))
        X_batch_them = cp_sparse.csr_matrix(matrix_them_cpu[batch_idx].astype(np.float32))
        y_batch = cp.asarray(labels_cpu[batch_idx]).reshape(-1, 1)

        # Forward (2 sparse matmuls séparées)
        activations, zs = forward_pass(X_batch_us, X_batch_them, weights, biases)
        A4 = activations[-1]

        # Loss (BCE fusionnée)
        loss = cp.mean(bce_loss_fused(y_batch, A4))
        total_loss += loss

        # Backward (utilise .T implicite CSC, PAS de conversion CSR)
        dWs, dbs = backward_pass(X_batch_us, X_batch_them, y_batch, activations, zs, weights)
        
        dWs = clip_gradients(dWs)
        dbs = clip_gradients(dbs)
        
        # ==========================================================
        # ADAM DENSE POUR TOUTES LES COUCHES
        # ==========================================================
        for j in range(len(weights)):
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
        
        
        elapsed = time.perf_counter() - t0
        # print(f"Adam 2,3,4 | Epoch {epoch}, Batch {i//BATCH_SIZE} | Temps: {elapsed:.2f}s | Durée: {(elapsed - tpast)*1000:.1f}ms")
        # tpast = elapsed
        # Affichage réduit (tous les 200 batches pour éviter la synchro GPU)
        batch_num = i // BATCH_SIZE
        if batch_num % 200 == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0
            loss_val = float(loss)
            Z1_us = zs[0]
            alive_mask = (Z1_us > 0) & (Z1_us <= 1.0)
            alive_ratio = float(cp.mean(alive_mask))
            msg = f"Epoch {epoch}, Batch {batch_num} | Loss: {loss_val:.4f} | Neurones Actifs C1: {alive_ratio:.1%} | Temps: {elapsed:.2f}s | Durée: {(elapsed - tpast)*1000:.1f}ms"
            print(msg)
            tpast = elapsed

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"--- Fin Epoch {epoch}, Moyenne Loss: {total_loss / (len(indices)//BATCH_SIZE):.4f}, Temps écoulé: {elapsed:.2f}s ---")

cp.savez('model_halfKP_V4.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
print("Modèle sauvegardé sous 'model_halfKP_V4.npz'")