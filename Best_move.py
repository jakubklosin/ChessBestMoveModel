import chess
import chess.engine
from analyze_Board import get_fen_from_eval_pola

# FEN z obrazów
fen = get_fen_from_eval_pola()
print("Pozycja FEN:", fen)

# Silnik
engine_path = "/opt/homebrew/bin/stockfish"
board = chess.Board(fen)

with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
    result = engine.play(board, chess.engine.Limit(time=1))
    move = result.move

    print("\nNajlepszy ruch (notacja):", move)

    # Tłumaczenie ruchu na język naturalny
    piece = board.piece_at(move.from_square)
    piece_name = {
        chess.PAWN: "pionkiem",
        chess.KNIGHT: "skoczkiem",
        chess.BISHOP: "gońcem",
        chess.ROOK: "wieżą",
        chess.QUEEN: "hetmanem",
        chess.KING: "królem"
    }.get(piece.piece_type, "figurą")

    to_square = chess.square_name(move.to_square)
    print(f"Najlepszy ruch (opis): zagraj {piece_name} na pole {to_square}")
