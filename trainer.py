# IMPORTS
import numpy as np
import chess
import pandas as pd

print("Imports done")

# DATASET
raw_data = pd.read_csv('archive/games.csv')

win = raw_data["winner"].to_numpy()
win_to_int = np.vectorize(lambda x: 1.0 if x == 'white' else (0.0 if x == 'black' else 0.5))
labels = win_to_int(win)

def format_games(moves):
    boards = []
    board = chess.Board()
    for move in moves.split():
        # 1. On regarde à qui c'est le tour AVANT de jouer
        turn = board.turn 
        
        input_vector = np.zeros(64*12, dtype=np.float32) # Moins de RAM !
        
        for square in chess.SQUARES:
            display_square = square if turn == chess.WHITE else chess.square_mirror(square)
            piece = board.piece_at(square)
            
            if piece is not None:
                piece_type = piece.piece_type - 1
                if turn == chess.WHITE:
                    color_offset = 0 if piece.color == chess.WHITE else 6
                else:
                    color_offset = 0 if piece.color == chess.BLACK else 6               
                     
                input_vector[display_square * 12 + color_offset + piece_type] = 1                
        boards.append(input_vector)
        
        # 2. On joue le coup SEULEMENT APRÈS avoir photographié le plateau
        board.push_san(move)
        
    # 3. On coupe les 20 premiers demi-coups (10 premiers coups) de l'ouverture
    input_dim = 64 * 12
    if len(boards) == 0:
        return np.empty((0, input_dim), dtype=np.float32)
    arr = np.array(boards, dtype=np.float32)
    return arr[20:] if arr.shape[0] > 20 else np.empty((0, input_dim), dtype=np.float32)


format_games_array = np.vectorize(format_games)
moves = raw_data["moves"].to_numpy()
# print(format_games(moves[0]))

# Remplacement : ne pas utiliser np.vectorize pour des listes de longueurs variables
boards_list = [format_games(m) for m in moves]   # liste de tableaux numpy (une par partie)
# Filtrer les tableaux vides avant la concaténation
non_empty = [b for b in boards_list if b.shape[0] > 0]
if len(non_empty) == 0:
    BOARDS = np.empty((0, 64 * 12), dtype=np.float32)
else:
    BOARDS = np.concatenate(non_empty)              # (total_positions, 768)
counts = [len(b) for b in boards_list]

# 1. On crée d'abord le répétition classique (comme tu faisais avant)
lbls = np.repeat(labels, counts)

# 2. On génère un tableau d'alternance (0, 1, 0, 1...) pour chaque partie
# On peut le faire en créant une suite d'indices pour chaque position dans sa partie
position_indices = np.concatenate([np.arange(c) for c in counts])

# 3. On applique l'inversion : si l'indice est impair (coup des noirs), on fait 1 - label
# (indices % 2) vaut 1 pour les noirs, 0 pour les blancs
LABELS = np.where(position_indices % 2 == 1, 1.0 - lbls, lbls)

print("Dataset loaded and formatted")

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


# WEIGHTS INITIALIZATION
W1 = np.random.randn(INPUT_COUNT, NEURONE_COUNT) * np.sqrt(2.0 / INPUT_COUNT)
b1 = np.zeros((1, NEURONE_COUNT))
W2 = np.random.randn(NEURONE_COUNT, 1) * np.sqrt(1.0 / NEURONE_COUNT)
b2 = np.zeros((1, 1))

def init_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# FORWARD PASS
def forward_pass(input, W1, b1, W2, b2):
    
    # HIDDEN LAYER
    Z1 = np.dot(input,W1) + b1
    A1 = ReLU(Z1)

    # OUTPUT LAYER
    Z2 = np.dot(A1 , W2) + b2
    A2 = Sigmoid(Z2)
    return A1, Z1, A2

# BACKWARD PASS
def backward_pass(input, W1, b1, A1,Z1, W2, b2, A2, labels, alpha):
    m = input.shape[0]
    
    # d2 = 2 * ( A2 - labels ) * Sigmoid_derivative(A2)    
    # Remplacement de MSE par Binary Cross Entropy (BCE)
    # La dérivée de BCE + Sigmoid se simplifie en (A2 - labels)
    d2 = (A2 - labels)
    # d2 = 2 * ( A2 - labels ) * Sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, d2) / m
    db2 = np.sum(d2, axis=0, keepdims=True) / m

    d1 = np.dot(d2, W2.T) * ReLU_derivative(Z1)
    dW1 = np.dot(input.T, d1) / m  
    db1 = np.sum(d1, axis=0, keepdims=True) / m

    b2 = b2 - alpha * db2
    W2 = W2 - alpha * dW2
    b1 = b1 - alpha * db1
    W1 = W1 - alpha * dW1
    return W1, b1, W2, b2

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
