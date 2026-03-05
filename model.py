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
        for move in sort_moves(board):
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
        for move in sort_moves(board):
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
    
def score_move(board : chess.Board, move : chess.Move):
    
    if board.is_capture(move) :
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim:
            return 9000 + (victim.piece_type * 100 ) - attacker.piece_type
        return 9000
    
    if move.promotion:
        return 8000
    
    return 5000
    # if 
    
def sort_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(key= lambda x : score_move(board, x), reverse=True)
    return moves

    
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
    