import pandas as pd
import numpy as np
from scipy import sparse
import time
import src.chess_nnue.config as config

CHUNK_SIZE = 500_000
SCALING_FACTOR = 75
HALFKP_DIM = 64 * 640          # 40 960 features

# --- LOOKUP TABLE HalfKP ---
# piece_char -> (piece_type 0-4, is_white, is_king)
_HP = {}
for _i, _c in enumerate('PNBRQ'):
    _HP[_c]         = (_i, True,  False)
    _HP[_c.lower()] = (_i, False, False)
_HP['K'] = (-1, True,  True)
_HP['k'] = (-1, False, True)

def process_chunk_halfkp(fens, evals):
    """
    Parsing FEN manuel + encodage HalfKP des DEUX perspectives + labels vectorisés.
    Retourne (mat_white, mat_black), labels.
    Aucune dépendance à python-chess → ~10x plus rapide.
    """
    n = len(fens)
    rows_us, cols_us = [], []     # Remplacera w
    rows_them, cols_them = [], [] # Remplacera b
    vals = np.empty(n, dtype=np.float64)
    white_turn = np.empty(n, dtype=np.bool_)

    for i in range(n):
        fen = fens[i]
        ev  = str(evals[i])

        # --- Tour ---
        sp   = fen.index(' ')
        white_turn[i] = fen[sp + 1] == 'w'

        # --- Eval ---
        if '#' in ev:
            vals[i] = 10000.0 if '+' in ev else -10000.0
        else:
            try:
                vals[i] = int(ev)
            except ValueError:
                cleaned = ''.join(c for c in ev if c.isdigit() or c == '-')
                vals[i] = int(cleaned) if cleaned and cleaned != '-' else 0

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
    labels = (1.0 / (1.0 + np.exp(-vals / SCALING_FACTOR))).astype(np.float32)

    ones_us = np.ones(len(rows_us), dtype=np.float32)
    ones_them = np.ones(len(rows_them), dtype=np.float32)
    mat_us  = sparse.csr_matrix((ones_us, (rows_us, cols_us)), shape=(n, HALFKP_DIM), dtype=np.float32)
    mat_them  = sparse.csr_matrix((ones_them, (rows_them, cols_them)), shape=(n, HALFKP_DIM), dtype=np.float32)
    return (mat_us, mat_them), labels

# --- BOUCLE PRINCIPALE ---
t0 = time.perf_counter()
all_sparse_w = []
all_sparse_b = []
all_labels   = []
total = 0

print("Début du traitement HalfKP optimisé (deux perspectives)...")
reader = pd.read_csv(config.RAW_DATASET_DIR, chunksize=CHUNK_SIZE)

for chunk in reader:
    (mat_us, mat_them), labels = process_chunk_halfkp(chunk['FEN'].values, chunk['Evaluation'].values)
    all_sparse_w.append(mat_us)
    all_sparse_b.append(mat_them)
    all_labels.append(labels)
    total += len(labels)
    elapsed = time.perf_counter() - t0
    print(f"  {total:>10,} positions  |  {elapsed:.1f}s  |  {total/elapsed:,.0f} pos/s")

# --- Assemblage final & sauvegarde ---
print("Assemblage final...")
final_moves_us = sparse.vstack(all_sparse_w, format='csr')
final_moves_them = sparse.vstack(all_sparse_b, format='csr')
final_labels  = np.concatenate(all_labels)

sparse.save_npz(config.MOVES_US_DIR, final_moves_us)
sparse.save_npz(config.MOVES_THEM_DIR, final_moves_them)
np.savez(config.LABELS_DIR, Y=final_labels)

dt = time.perf_counter() - t0
print(f"\nTerminé ! {final_moves_us.shape[0]:,} positions ({HALFKP_DIM} features) en {dt:.1f}s ({final_moves_us.shape[0]/dt:,.0f} pos/s)")
