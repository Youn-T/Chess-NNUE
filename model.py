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

# --- Table de Transposition (Hash Table) ---
# Format : { hash: (depth, score, flag, best_move) }
# flags : 0 = EXACT, 1 = ALPHA (borne sup), 2 = BETA (borne inf)
transposition_table = {}

# --- Killer Moves ---
# On stocke 2 coups par niveau de profondeur (ply)
MAX_PLY = 64
killer_moves = [[None, None] for _ in range(MAX_PLY)]

# --- History Heuristic ---
# [couleur][case_depart][case_arrivee]
history_scores = np.zeros((2, 64, 64), dtype=int)

def minimax(board, depth, alpha=float('-inf'), beta=float('inf'), ply=0):
    original_alpha = alpha
    board_hash = board.fen() # python-chess génère ça très vite
    
    # --- 1. CONSULTATION DE LA TABLE DE TRANSPOSITION ---
    tt_entry = transposition_table.get(board_hash)
    hash_move = None
    if tt_entry:
        tt_depth, tt_score, tt_flag, tt_move = tt_entry
        hash_move = tt_move
        if tt_depth >= depth:
            if tt_flag == 0: # EXACT
                return tt_score, tt_move
            elif tt_flag == 1: # ALPHA
                beta = min(beta, tt_score)
            elif tt_flag == 2: # BETA
                alpha = max(alpha, tt_score)
            
            if alpha >= beta:
                return tt_score, tt_move

    # --- 2. CONDITION D'ARRÊT ---
    if board.is_game_over() or depth == 0:
        # (Utilisez votre fonction predict_fast optimisée ici)
        us_idx, them_idx = get_active_indices(board)
        score = predict_fast(us_idx, them_idx, weights, biases)
        if board.turn == chess.BLACK: score = 1.0 - score
        return score, None

    # --- 3. TRI DES COUPS ---
    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: get_move_priority(board, m, depth, ply, hash_move), reverse=True)

    best_move = None
    best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')

    for move in legal_moves:
        board.push(move)
        eval_val, _ = minimax(board, depth - 1, alpha, beta, ply + 1)
        board.pop()

        if board.turn == chess.WHITE:
            if eval_val > best_eval:
                best_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
        else:
            if eval_val < best_eval:
                best_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)

        # --- 4. COUPURE BETA (L'endroit où on apprend) ---
        if alpha >= beta:
            # On stocke le Killer Move (si ce n'est pas une capture)
            if not board.is_capture(move):
                if move != killer_moves[ply][0]:
                    killer_moves[ply][1] = killer_moves[ply][0]
                    killer_moves[ply][0] = move
                
                # On incrémente l'History Heuristic
                color = int(board.turn)
                history_scores[color][move.from_square][move.to_square] += depth * depth
            
            break # Coupure Alpha-Beta

    # --- 5. SAUVEGARDE DANS LA TABLE DE TRANSPOSITION ---
    # Déterminer le flag (Exact, Borne Inf ou Borne Sup)
    if best_eval <= original_alpha:
        flag = 1 # ALPHA (Upper bound)
    elif best_eval >= beta:
        flag = 2 # BETA (Lower bound)
    else:
        flag = 0 # EXACT
        
    transposition_table[board_hash] = (depth, best_eval, flag, best_move)
    
    return best_eval, best_move
    
def get_move_priority(board, move, depth, ply, hash_move):
    # 1. Le coup de la Table de Transposition (Priorité absolue)
    if move == hash_move:
        return 1000000
    
    # 2. Captures (MVV-LVA)
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        vic_val = victim.piece_type if victim else 1 # En-passant
        return 900000 + (vic_val * 10) - attacker.piece_type

    # 3. Killer Moves (Coups calmes qui ont causé des coupures beta récemment)
    if move == killer_moves[ply][0]:
        return 800000
    if move == killer_moves[ply][1]:
        return 700000
    
    # 4. History Heuristic (Coups calmes qui fonctionnent globalement)
    color = int(board.turn)
    return history_scores[color][move.from_square][move.to_square]
    
def sort_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(key= lambda x : get_move_priority(board, x, 0, 0, None), reverse=True)
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
    