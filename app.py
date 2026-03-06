from flask import Flask,session,jsonify, render_template
import chess
import chess.svg
import model
import stockfish_model as st_model
from stockfish import Stockfish
st = Stockfish(path="C:/Users/yount/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
st.set_elo_rating(1600)
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using sessions

def board_to_matrice(board : chess.Board):
    matrix = [ [0] * 8 for _ in range(8) ]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = 7 - (square // 8)  # Convert to row index
            col = square % 8          # Convert to column index
            matrix[row][col] = chess.svg.piece(piece)  # Store the piece symbol in the matrix
    return matrix

def html_id_to_square(html_id):
    html_id = int(html_id)
    return (7 - html_id // 8) * 8 + (html_id % 8)

def square_to_html_id(square):
    return (7 - square // 8) * 8 + (square % 8)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ai")
def ai():
    return render_template("ai.html")

@app.route("/init_game")
def init_game():
    board = chess.Board()
    session['board'] = board.fen()  # Store the board state in the session
    return jsonify({"message": "Game initialized", "board": board_to_matrice(board)})

@app.route("/get_legal_moves/<piece_id>")
def get_legal_moves(piece_id):
    board = chess.Board(session.get('board', chess.STARTING_FEN))  # Retrieve the board state from the session
    piece_square = html_id_to_square(piece_id)  # Convert HTML id to square index
    legal_moves = [[square_to_html_id(move.from_square), square_to_html_id(move.to_square)] for move in board.legal_moves if move.from_square == piece_square]
    return jsonify({"legal_moves": legal_moves})

@app.route("/make_move/<from_square_id>/<to_square_id>")
def make_move(from_square_id, to_square_id): 
    board = chess.Board(session.get('board', chess.STARTING_FEN))  # Retrieve the board state from the session
    from_square = html_id_to_square(from_square_id)
    to_square = html_id_to_square(to_square_id)
    
    # Trouver le coup correspondant parmi les coups légaux (gère les promotions automatiquement)
    move = next((m for m in board.legal_moves if m.from_square == from_square and m.to_square == to_square), None)
    
    if move:
        board.push(move)  # Make the move on the board
        if board.is_game_over():
            result = board.result()
            session['board'] = chess.Board().fen()  # Reset the board state in the session
            return jsonify({"message": f"Game over: {result}", "board": board_to_matrice(board)})
        session['board'] = board.fen()  # Update the board state in the session
        return jsonify({"message": "Move made", "board": board_to_matrice(board)})
    else:
        return jsonify({"message": "Illegal move"}), 400
    
@app.route("/stockfish_move")
def stockfish_move():
    board = chess.Board(session.get('board', chess.STARTING_FEN))  # Retrieve the board state from the session
    
    if board.turn == chess.WHITE: 
        ## IA joue
        # _, best_move = st_model.minimax(board, depth=4)  # Ajustez la profondeur selon vos besoins

        st.set_fen_position(board.fen())
        best_move = st.get_best_move()
        board.push(chess.Move.from_uci(best_move))  # L'IA joue son meilleur coup    
        if board.is_game_over():
            result = board.result()
            session['board'] = chess.Board().fen()  # Reset the board state in the session
            return jsonify({"message": f"Game over: {result}", "board": board_to_matrice(board)})
        
        session['board'] = board.fen()  # Update the board state in the session
        return jsonify({"message": "Move made", "board": board_to_matrice(board)})
    else:
        return jsonify({"message": "Not AI turn"}), 400

@app.route("/ai_move")
def ai_move():
    board = chess.Board(session.get('board', chess.STARTING_FEN))  # Retrieve the board state from the session
    
    if board.turn == chess.BLACK: 
        ## IA joue
        _, best_move, acc = model.minimax(board, depth=7)  # Ajustez la profondeur selon vos besoins
        board.push(best_move)  # L'IA joue son meilleur coup    
        if board.is_game_over():
            result = board.result()
            session['board'] = chess.Board().fen()  # Reset the board state in the session
            return jsonify({"message": f"Game over: {result}", "board": board_to_matrice(board)})
        
        session['board'] = board.fen()  # Update the board state in the session
        return jsonify({"message": "Move made", "board": board_to_matrice(board)})
    else:
        return jsonify({"message": "Not AI turn"}), 400


app.run(debug=True)