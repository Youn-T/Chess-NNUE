import pandas as pd
import numpy as np
from scipy import sparse
import time
from pathlib import Path

# --- CONFIGURATION ---
# Résolution robuste du chemin vers `chessData.csv` (dossier parent du script)
BASE_DIR = Path(__file__).resolve().parent.parent
_candidate = BASE_DIR / 'chessData.csv'
if not _candidate.exists():
    _candidate = Path.cwd() / 'chessData.csv'
if not _candidate.exists():
    raise FileNotFoundError(
        f"chessData.csv introuvable. Recherché: {BASE_DIR / 'chessData.csv'} and {Path.cwd() / 'chessData.csv'}"
    )
FILE_PATH = str(_candidate)

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
    Parsing FEN manuel + encodage HalfKP + labels vectorisés.
    Aucune dépendance à python-chess → ~10x plus rapide.
    """
    n = len(fens)
    rows, cols = [], []
    vals = np.empty(n, dtype=np.float64)
    white_turn = np.empty(n, dtype=np.bool_)

    for i in range(n):
        fen = fens[i]
        ev  = str(evals[i])

        # --- Tour ---
        sp   = fen.index(' ')
        is_w = fen[sp + 1] == 'w'
        white_turn[i] = is_w

        # --- Eval ---
        if '#' in ev:
            vals[i] = 10000.0 if '+' in ev else -10000.0
        else:
            try:
                vals[i] = int(ev)
            except ValueError:
                cleaned = ''.join(c for c in ev if c.isdigit() or c == '-')
                vals[i] = int(cleaned) if cleaned and cleaned != '-' else 0

        # --- Passe 1 : trouver le Roi du joueur actif ---
        king_sq = -1
        sq = 56
        for ch in fen[:sp]:
            if ch == '/':
                sq -= 16
            elif '1' <= ch <= '8':
                sq += ord(ch) - 48
            else:
                info = _HP[ch]
                if info[2] and info[1] == is_w:        # c'est un roi de la bonne couleur
                    king_sq = sq if is_w else (sq ^ 56) # miroir si noirs
                sq += 1

        # --- Passe 2 : encoder les pièces (sauf les rois) ---
        king_base = king_sq * 640                       # pré-calcul offset roi
        sq = 56
        for ch in fen[:sp]:
            if ch == '/':
                sq -= 16
            elif '1' <= ch <= '8':
                sq += ord(ch) - 48
            else:
                pt, iw, is_king = _HP[ch]
                if not is_king:
                    dsq = sq if is_w else (sq ^ 56)
                    co  = 0 if (iw == is_w) else 5      # 5 types de pièces par couleur
                    rows.append(i)
                    cols.append(king_base + (co + pt) * 64 + dsq)
                sq += 1

    # --- Labels vectorisés ---
    vals[~white_turn] *= -1
    labels = (1.0 / (1.0 + np.exp(-vals / SCALING_FACTOR))).astype(np.float32)

    data = np.ones(len(rows), dtype=np.float32)
    mat  = sparse.csr_matrix((data, (rows, cols)),
                             shape=(n, HALFKP_DIM), dtype=np.float32)
    return mat, labels

# --- BOUCLE PRINCIPALE ---
t0 = time.perf_counter()
all_sparse = []
all_labels = []
total = 0

print("Début du traitement HalfKP optimisé...")
reader = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)

for chunk in reader:
    mat, labels = process_chunk_halfkp(chunk['FEN'].values, chunk['Evaluation'].values)
    all_sparse.append(mat)
    all_labels.append(labels)
    total += len(labels)
    elapsed = time.perf_counter() - t0
    print(f"  {total:>10,} positions  |  {elapsed:.1f}s  |  {total/elapsed:,.0f} pos/s")

# --- Assemblage final & sauvegarde ---
print("Assemblage final...")
final_moves  = sparse.vstack(all_sparse, format='csr')
final_labels = np.concatenate(all_labels)

sparse.save_npz('moves_halfKP_V1.npz', final_moves)
np.savez('labels_halfKP_V1.npz', Y=final_labels)

dt = time.perf_counter() - t0
print(f"\nTerminé ! {final_moves.shape[0]:,} positions ({HALFKP_DIM} features) en {dt:.1f}s ({final_moves.shape[0]/dt:,.0f} pos/s)")
