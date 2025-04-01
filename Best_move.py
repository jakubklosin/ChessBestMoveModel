import chess
import chess.engine

fen = "KppQpKNp/pPRppRRB/RpBPRppR/ppqRpppp/qqpppppp/ppppqrpp/rnpppqqq/1ppk1pbp w - - 0 1"

# Ścieżka do silnika Stockfish
engine_path = "/opt/homebrew/bin/stockfish"

# Ustawienie pozycji i uruchomienie silnika
board = chess.Board(fen)
with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
    result = engine.play(board, chess.engine.Limit(time=1))
    print("Najlepszy ruch:", result.move)
