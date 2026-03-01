# # IMPORTS
# import numpy as np
# import chess
# import pandas as pd

# print("Imports done")

# MINIMUM_RATING = 1800
# FIRST_N_MOVES_TO_SKIP = 10

# # DATASET
# raw_data = pd.read_csv('archive/games.csv')
# raw_data = raw_data.loc[(raw_data['white_rating'] > MINIMUM_RATING) & (raw_data['black_rating'] > MINIMUM_RATING)]
# win = raw_data["winner"].to_numpy()
# win_to_int = np.vectorize(lambda x: 1.0 if x == 'white' else (0.0 if x == 'black' else 0.5))
# labels = win_to_int(win)

# def format_games(moves):
#     boards = []
#     board = chess.Board()
#     for move in moves.split():
#         # 1. On regarde à qui c'est le tour AVANT de jouer
        # turn = board.turn 
        
        # input_vector = np.zeros(64*12, dtype=np.float32) # Moins de RAM !
        
        # for square in chess.SQUARES:
        #     display_square = square if turn == chess.WHITE else chess.square_mirror(square)
        #     piece = board.piece_at(square)
            
        #     if piece is not None:
        #         piece_type = piece.piece_type - 1
        #         if turn == chess.WHITE:
        #             color_offset = 0 if piece.color == chess.WHITE else 6
        #         else:
        #             color_offset = 0 if piece.color == chess.BLACK else 6               
                     
        #         input_vector[display_square * 12 + color_offset + piece_type] = 1                
#         boards.append(input_vector)
        
#         # 2. On joue le coup SEULEMENT APRÈS avoir photographié le plateau
#         board.push_san(move)
        
#     # 3. On coupe les 20 premiers demi-coups (10 premiers coups) de l'ouverture
#     input_dim = 64 * 12
#     if len(boards) == 0:
#         return np.empty((0, input_dim), dtype=np.float32)
#     arr = np.array(boards, dtype=np.float32)
#     return arr[FIRST_N_MOVES_TO_SKIP:] if arr.shape[0] > FIRST_N_MOVES_TO_SKIP else np.empty((0, input_dim), dtype=np.float32)


# format_games_array = np.vectorize(format_games)
# moves = raw_data["moves"].to_numpy()
# # print(format_games(moves[0]))

# # Remplacement : ne pas utiliser np.vectorize pour des listes de longueurs variables
# boards_list = [format_games(m) for m in moves]   # liste de tableaux numpy (une par partie)
# # Filtrer les tableaux vides avant la concaténation
# non_empty = [b for b in boards_list if b.shape[0] > 0]
# if len(non_empty) == 0:
#     BOARDS = np.empty((0, 64 * 12), dtype=np.float32)
# else:
#     BOARDS = np.concatenate(non_empty)              # (total_positions, 768)
# counts = [len(b) for b in boards_list]

# # 1. On crée d'abord le répétition classique (comme tu faisais avant)
# lbls = np.repeat(labels, counts)

# # 2. On génère un tableau d'alternance (0, 1, 0, 1...) pour chaque partie
# # On peut le faire en créant une suite d'indices pour chaque position dans sa partie
# position_indices = np.concatenate([np.arange(c) for c in counts])

# # 3. On applique l'inversion : si l'indice est impair (coup des noirs), on fait 1 - label
# # (indices % 2) vaut 1 pour les noirs, 0 pour les blancs
# LABELS = np.where(position_indices % 2 == 1, 1.0 - lbls, lbls)

# print("Dataset loaded and formatted")

# # HYPERPARAMETERS
# NEURONE_COUNT = 512
# INPUT_COUNT = 768
# BATCH_SIZE = 512
# LEARNING_RATE = 0.05
# EPOCHS = 10

# # KEY FUNCTIONS
# def ReLU(x):
#     return np.maximum(0, x)
    
# def CReLU(x):
#     return np.clip(x, 0, 1)

# def ReLU_derivative(x):
#     return np.where(x > 0, 1, 0)

# def CReLU_derivative(x):
#     return np.where((x > 0) & (x < 1), 1, 0)

# def Sigmoid(x):
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + np.exp(-x))

# def Sigmoid_derivative(x):
#     return x * (1 - x)


# def MSE(prediction, target):
#     return np.square( prediction - target )


# # WEIGHTS INITIALIZATION
# W1 = np.random.randn(INPUT_COUNT, NEURONE_COUNT) * np.sqrt(2.0 / INPUT_COUNT)
# b1 = np.zeros((1, NEURONE_COUNT))
# W2 = np.random.randn(NEURONE_COUNT, 1) * np.sqrt(1.0 / NEURONE_COUNT)
# b2 = np.zeros((1, 1))

# def init_weights(input_size, hidden_size, output_size):
#     W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
#     b1 = np.zeros((1, hidden_size))
#     W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
#     b2 = np.zeros((1, output_size))
#     return W1, b1, W2, b2


# # FORWARD PASS
# def forward_pass(input, W1, b1, W2, b2):
    
#     # HIDDEN LAYER
#     Z1 = np.dot(input,W1) + b1
#     A1 = ReLU(Z1)

#     # OUTPUT LAYER
#     Z2 = np.dot(A1 , W2) + b2
#     A2 = Sigmoid(Z2)
#     return A1, Z1, A2

# # BACKWARD PASS
# def backward_pass(input, W1, b1, A1,Z1, W2, b2, A2, labels, alpha):
#     m = input.shape[0]
    
#     # d2 = 2 * ( A2 - labels ) * Sigmoid_derivative(A2)    
#     # Remplacement de MSE par Binary Cross Entropy (BCE)
#     # La dérivée de BCE + Sigmoid se simplifie en (A2 - labels)
#     d2 = (A2 - labels)
#     # d2 = 2 * ( A2 - labels ) * Sigmoid_derivative(A2)
#     dW2 = np.dot(A1.T, d2) / m
#     db2 = np.sum(d2, axis=0, keepdims=True) / m

#     d1 = np.dot(d2, W2.T) * ReLU_derivative(Z1)
#     dW1 = np.dot(input.T, d1) / m  
#     db1 = np.sum(d1, axis=0, keepdims=True) / m

#     b2 = b2 - alpha * db2
#     W2 = W2 - alpha * dW2
#     b1 = b1 - alpha * db1
#     W1 = W1 - alpha * dW1
#     return W1, b1, W2, b2

# print("Functions defined, starting training...")
# W1, b1, W2, b2 = init_weights(INPUT_COUNT, NEURONE_COUNT, 1)
# for epoch in range(EPOCHS):
#     indices = np.arange(len(BOARDS))
#     np.random.shuffle(indices)
    
#     BOARDS_SHUFFLED = BOARDS[indices]
#     LABELS_SHUFFLED = LABELS[indices]
    
#     for i in range(0, len(BOARDS), BATCH_SIZE):
#         if (i+BATCH_SIZE) > len(BOARDS):
#             break
#         X_batch = BOARDS_SHUFFLED[i:i+BATCH_SIZE]
#         y_batch = LABELS_SHUFFLED[i:i+BATCH_SIZE].reshape(-1, 1)
        
#         A1, Z1, A2 = forward_pass(X_batch, W1, b1, W2, b2)
        
#         rounded_predictions = np.round(A2.flatten())
#         # accuracy = np.mean(rounded_predictions == y_batch.flatten())
#         epsilon = 1e-15
#         A2_clipped = np.clip(A2, epsilon, 1 - epsilon)

#         # Calcul de la Binary Cross Entropy
#         loss = -np.mean(y_batch * np.log(A2_clipped) + (1 - y_batch) * np.log(1 - A2_clipped))

#         print(f"Epoch {epoch}, Batch {i//BATCH_SIZE}: Loss: {loss}")
        
#         W1, b1, W2, b2 = backward_pass(X_batch, W1, b1, A1,Z1, W2, b2, A2, y_batch, LEARNING_RATE)
        
#     print(f"End of Epoch {epoch}")
    
# with open("model_weights.npz", "wb") as f:
#     np.savez(f, W1=W1, b1=b1, W2=W2, b2=b2)
import numpy as np
import chess
import pandas as pd

# --- CONFIGURATION ---
MINIMUM_RATING = 1800
MOVES_TO_SKIP = 20  # On saute l'ouverture pour éviter le bruit
NEURONE_COUNT = 512
INPUT_COUNT = 768
BATCH_SIZE = 1024
LEARNING_RATE = 0.1
MOMENTUM = 0.9  # Aide à sortir du plafond de 0.69
EPOCHS = 15

print("Chargement des données...")
raw_data = pd.read_csv('archive/games.csv')
raw_data = raw_data.loc[(raw_data['white_rating'] > MINIMUM_RATING) & (raw_data['black_rating'] > MINIMUM_RATING)]

def format_games(moves, winner_val):
    game_data = []
    board = chess.Board()
    
    for move_san in moves.split():
        turn = board.turn # True = Blanc, False = Noir
        
        # Photo du plateau AVANT le coup
        input_vector = np.zeros(768, dtype=np.float32)
        for square, piece in board.piece_map().items():
            # Inversion si c'est aux noirs
            d_square = square if turn == chess.WHITE else chess.square_mirror(square)
            
            # Couleur relative : 0-5 pour "Moi", 6-11 pour "Lui"
            if turn == chess.WHITE:
                color_offset = 0 if piece.color == chess.WHITE else 6
            else:
                color_offset = 0 if piece.color == chess.BLACK else 6
            
            index = d_square * 12 + color_offset + (piece.piece_type - 1)
            input_vector[index] = 1
            
        # Label relatif : est-ce que celui qui a le trait gagne ?
        # winner_val: 1=Blanc gagne, 0=Noir gagne, 0.5=Nul
        rel_label = winner_val if turn == chess.WHITE else 1.0 - winner_val
        
        game_data.append((input_vector, rel_label))
        
        try:
            board.push_san(move_san)
        except: break
            
    return game_data[MOVES_TO_SKIP:] # On ignore le début de partie

# --- PRÉPARATION DU DATASET ---
all_positions = []
all_labels = []

win_map = {'white': 1.0, 'black': 0.0, 'draw': 0.5}
for moves_str, winner in zip(raw_data['moves'], raw_data['winner']):
    formatted = format_games(moves_str, win_map[winner])
    for vec, lbl in formatted:
        all_positions.append(vec)
        all_labels.append(lbl)

X = np.array(all_positions)
y = np.array(all_labels).reshape(-1, 1)
print(f"Dataset prêt : {X.shape[0]} positions.")

# --- INITIALISATION ---
def init_params():
    W1 = np.random.randn(INPUT_COUNT, NEURONE_COUNT) * np.sqrt(2.0 / INPUT_COUNT)
    b1 = np.zeros((1, NEURONE_COUNT))
    W2 = np.random.randn(NEURONE_COUNT, 1) * np.sqrt(1.0 / NEURONE_COUNT)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()

# Variables de Momentum (vitesse)
vW1, vb1 = np.zeros_like(W1), np.zeros_like(b1)
vW2, vb2 = np.zeros_like(W2), np.zeros_like(b2)

# --- FONCTIONS DU RÉSEAU ---
def ReLU(x): return np.maximum(0, x)
def ReLU_der(x): return (x > 0).astype(float)
def Sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Sigmoid(Z2)
    return Z1, A1, Z2, A2

# --- BOUCLE D'ENTRAÎNEMENT ---
for epoch in range(EPOCHS):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    total_loss = 0
    
    for i in range(0, len(X), BATCH_SIZE):
        batch_X = X_shuffled[i:i+BATCH_SIZE]
        batch_y = y_shuffled[i:i+BATCH_SIZE]
        m = batch_X.shape[0]
        
        # Forward
        Z1, A1, Z2, A2 = forward(batch_X, W1, b1, W2, b2)
        
        # Calcul de la Loss (BCE)
        loss = -np.mean(batch_y * np.log(A2 + 1e-15) + (1 - batch_y) * np.log(1 - A2 + 1e-15))
        total_loss += loss
        
        # Backprop
        dZ2 = A2 - batch_y
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * ReLU_der(Z1)
        dW1 = np.dot(batch_X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update avec MOMENTUM
        vW1 = MOMENTUM * vW1 - LEARNING_RATE * dW1
        vb1 = MOMENTUM * vb1 - LEARNING_RATE * db1
        vW2 = MOMENTUM * vW2 - LEARNING_RATE * dW2
        vb2 = MOMENTUM * vb2 - LEARNING_RATE * db2
        
        W1 += vW1; b1 += vb1; W2 += vW2; b2 += vb2
        
    avg_loss = total_loss / (len(X) // BATCH_SIZE)
    print(f"Époque {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Sauvegarde finale
np.savez("chess_model_v2.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Modèle sauvegardé !")