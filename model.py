import chess
import numpy as np
from scipy import sparse

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

def Leaky_Clipped_ReLU(x):
    # On applique la pente leaky en bas et on coupe à 1.0 en haut
    return np.where(x > 1.0, 1.0, np.where(x > 0, x, x * ALPHA_LEAKY))

def Leaky_Clipped_ReLU_derivative(x):
    # 1.0 entre 0 et 1, ALPHA en dessous de 0, et 0 au-dessus de 1
    grad = np.where((x >= 0) & (x <= 1.0), 1.0, 0.0)
    grad = np.where(x < 0, ALPHA_LEAKY, grad)
    return grad

def Leaky_ReLU(x):
    return np.where(x > 0, x, x * ALPHA_LEAKY)

def Leaky_ReLU_derivative(x):
    return np.where(x > 0, 1, ALPHA_LEAKY)

def Sigmoid(x):
    # Clip pour éviter les overflows d'exponentielle
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

model = np.load('model_halfKP_V4.npz')
W1 = model['W1']
W2 = model['W2']
W3 = model['W3']
W4 = model['W4']
b1 = model['b1']
b2 = model['b2']
b3 = model['b3']
b4 = model['b4']
weights = [W1, W2, W3, W4]
biases = [b1, b2, b3, b4]

def minimax(board, depth, alpha=float('-inf'), beta=float('inf')):
    # 1. Gérer les fins de partie de manière absolue
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None:
            return (0.5, None) # Match nul
        elif outcome.winner == chess.WHITE:
            return (1000 + depth, None) # Les blancs gagnent (on ajoute depth pour préférer les mats rapides)
        else:
            return (-1000 - depth, None) # Les noirs gagnent

    # 2. Évaluer la position avec le réseau de neurones
    if depth == 0:
        us_idx, them_idx = get_active_indices(board)
        prob = predict_fast(us_idx, them_idx, weights, biases)
        
        if board.turn == chess.BLACK:
            prob = 1.0 - prob
        return (prob, None)
    
    best_move = None
    
    # 3. Logique Minimax corrigée avec Alpha-Beta
    if board.turn == chess.WHITE:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_value, _ = minimax(board, depth - 1, alpha, beta)
            board.pop()
            
            if eval_value > max_eval:
                max_eval = eval_value
                best_move = move
                
            alpha = max(alpha, eval_value)
            if beta <= alpha:
                break # Coupure Alpha-Beta
        return (max_eval, best_move)
        
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_value, _ = minimax(board, depth - 1, alpha, beta)
            board.pop()
            
            if eval_value < min_eval:
                min_eval = eval_value
                best_move = move
                
            beta = min(beta, eval_value)
            if beta <= alpha:
                break # Coupure Alpha-Beta
        return (min_eval, best_move)
    
    
    
def get_active_indices(board):
    """
    Extrait directement les indices actifs depuis le plateau python-chess.
    Retourne deux listes d'entiers (us_indices, them_indices).
    """
    us_indices = []
    them_indices = []
    is_w = board.turn == chess.WHITE

    # 1. Trouver les rois
    wking_sq = board.king(chess.WHITE)
    bking_sq = board.king(chess.BLACK)
    
    # Si un roi manque (peut arriver dans des variantes ou tests), on gère ou on ignore.
    if wking_sq is None or bking_sq is None:
        return [], []

    wking_base = wking_sq * 640
    # Miroir vertical pour le roi noir
    bking_base = (bking_sq ^ 56) * 640 

    # 2. Parcourir les pièces
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
            
        iw = piece.color == chess.WHITE
        pt = piece.piece_type - 1 # Pion=0, Cavalier=1, etc.
        
        # Caractéristique point de vue Blanc
        feat_w = wking_base + ((0 if iw else 5) + pt) * 64 + sq
        # Caractéristique point de vue Noir (miroir vertical)
        feat_b = bking_base + ((0 if not iw else 5) + pt) * 64 + (sq ^ 56)
        
        if is_w:
            us_indices.append(feat_w)
            them_indices.append(feat_b)
        else:
            us_indices.append(feat_b)
            them_indices.append(feat_w)

    return us_indices, them_indices

def predict_fast(us_indices, them_indices, weights, biases):
    """
    Évaluation hyper-rapide par indexation directe (somme des lignes).
    Plus besoin de scipy.sparse.
    """
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases

    # Au lieu de multiplier une matrice creuse, on somme juste les lignes de W1
    # correspondant aux caractéristiques actives. C'est O(N_pieces).
    if len(us_indices) > 0:
        Z1_us = np.sum(W1[us_indices], axis=0) + b1
        Z1_them = np.sum(W1[them_indices], axis=0) + b1
    else:
        Z1_us = b1.copy()
        Z1_them = b1.copy()

    A1_us = Leaky_Clipped_ReLU(Z1_us)
    A1_them = Leaky_Clipped_ReLU(Z1_them)
    
    A1 = np.concatenate([A1_us, A1_them], axis=1) # concaténation 1D
    
    Z2 = np.dot(A1, W2) + b2
    A2 = Leaky_ReLU(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = Leaky_ReLU(Z3)
    
    Z4 = np.dot(A3, W4) + b4
    
    # Sigmoide simplifiée pour un seul scalaire
    x = max(min(Z4[0], 500), -500)
    return 1 / (1 + np.exp(-x))
    
## LEGACY
    
    
_HP = {}
for _i, _c in enumerate('PNBRQ'):
    _HP[_c]         = (_i, True,  False)
    _HP[_c.lower()] = (_i, False, False)
_HP['K'] = (-1, True,  True)
_HP['k'] = (-1, False, True)
  
HALFKP_DIM = 64 * 640 
def process_chunk_halfkp(fens):
    """
    Parsing FEN manuel + encodage HalfKP des DEUX perspectives + labels vectorisés.
    Retourne (mat_white, mat_black), labels.
    Aucune dépendance à python-chess → ~10x plus rapide.
    """
    n = len([fens.fen()])
    fens = [fens.fen()]
    rows_us, cols_us = [], []     # Remplacera w
    rows_them, cols_them = [], [] # Remplacera b
    vals = np.empty(n, dtype=np.float64)
    white_turn = np.empty(n, dtype=np.bool_)

    for i in range(1):
        fen = fens[i]

        # --- Tour ---
        sp   = fen.index(' ')
        white_turn[i] = fen[sp + 1] == 'w'  

        # --- Passe 1 : trouver les deux Rois ---
        wking_sq = bking_sq = -1
        sq = 56
        for ch in fen[:sp]:
            if ch == '/':
                sq -= 16
            elif '1' <= ch <= '8':
                sq += ord(ch) - 48
            else:
                info = _HP[ch]
                if info[2]:                             # is_king
                    if info[1]:                         # is_white
                        wking_sq = sq                   # perspective blanche : pas de miroir
                    else:
                        bking_sq = sq ^ 56              # perspective noire : miroir vertical
                sq += 1

        # --- Passe 2 : encoder les pièces (sauf les rois) pour les deux perspectives ---
        wking_base = wking_sq * 640
        bking_base = bking_sq * 640
        is_w = white_turn[i]

        sq = 56
        for ch in fen[:sp]:
            if ch == '/':
                sq -= 16
            elif '1' <= ch <= '8':
                sq += ord(ch) - 48
            else:
                pt, iw, is_king = _HP[ch]
                if not is_king:
                    # Caractéristique du point de vue Blanc
                    feat_w = wking_base + ((0 if iw else 5) + pt) * 64 + sq
                    # Caractéristique du point de vue Noir
                    feat_b = bking_base + ((0 if not iw else 5) + pt) * 64 + (sq ^ 56)
                    
                    # C'est ICI qu'on assigne Us et Them selon le trait !
                    if is_w: # Aux blancs de jouer
                        rows_us.append(i); cols_us.append(feat_w)
                        rows_them.append(i); cols_them.append(feat_b)
                    else:    # Aux noirs de jouer
                        rows_us.append(i); cols_us.append(feat_b)
                        rows_them.append(i); cols_them.append(feat_w)
                sq += 1

    # --- Labels vectorisés (du point de vue du joueur actif) ---
    vals[~white_turn] *= -1

    ones_us = np.ones(len(rows_us), dtype=np.float32)
    ones_them = np.ones(len(rows_them), dtype=np.float32)
    mat_us  = sparse.csr_matrix((ones_us, (rows_us, cols_us)), shape=(n, HALFKP_DIM), dtype=np.float32)
    mat_them  = sparse.csr_matrix((ones_them, (rows_them, cols_them)), shape=(n, HALFKP_DIM), dtype=np.float32)
    return (mat_us[0], mat_them[0])

# def board_to_input_vector(board : chess.Board, king_color = "black"):
#     input_w = np.zeros((1, INPUT_COUNT))
#     input_b = np.zeros((1, INPUT_COUNT))
    
#     king = board.king(chess.BLACK if king_color == "black" else chess.WHITE)
#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#         if piece is not None:
#             input_w[0][ king * 640 + ((0 if piece.color == chess.WHITE else 5) + piece.piece_type) * 64 + square ] = 1
#             input_b[0][ king * 640 + ((0 if piece.color == chess.BLACK else 5) + piece.piece_type) * 64 + (square ^ 56) ] = 1
#     return input_w, input_b

def predict(X_us, X_them, weights, biases):
    W1, W2, W3, W4 = weights
    b1, b2, b3, b4 = biases

    # 2 sparse matmuls séparées (évite vstack + overhead)
    Z1_us = X_us.dot(W1) + b1
    A1_us = Leaky_Clipped_ReLU(Z1_us)
    
    Z1_them = X_them.dot(W1) + b1
    A1_them = Leaky_Clipped_ReLU(Z1_them)
    
    A1 = np.concatenate([A1_us, A1_them], axis=1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = Leaky_ReLU(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = Leaky_ReLU(Z3)
    
    Z4 = np.dot(A3, W4) + b4
    A4 = Sigmoid(Z4)
    
    return A4[0]