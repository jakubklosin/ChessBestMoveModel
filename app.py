import streamlit as st
import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
import chess
import chess.engine
from analyze_Board import get_fen_from_eval_pola, process_board_image

# Konfiguracja
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
EVAL_DIR = "eval_pola"
RESIZED_DIR = "resized_images"

# Przekształcenia obrazu
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

st.title("Szachowy doradca ruchów ♖")
st.subheader("Nie wspomagaj się programem podczas grania prawdziwych partii")
side = st.radio("Z czyjej perspektywy analizujemy?", ["Białe", "Czarne"])
uploaded_file = st.file_uploader("Wgraj obraz szachownicy (screenshot)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Załadowana szachownica", use_column_width=True)

    # Zapisz tymczasowo
    os.makedirs(RESIZED_DIR, exist_ok=True)
    temp_name = "user_board.png"
    temp_path = os.path.join(RESIZED_DIR, temp_name)
    img.save(temp_path)

    # Zapytaj o FEN jeśli masz go już lub zgeneruj
    fen_input = st.text_input("Podaj FEN (opcjonalnie, inaczej zostanie rozpoznany ze zdjęcia)")

    if st.button("Analizuj pozycję"):
        if not fen_input:
            flip = side == "Czarne"
            process_board_image(img_name="user_board", input_dir=RESIZED_DIR, output_dir=EVAL_DIR, flip=flip)
            fen = get_fen_from_eval_pola(EVAL_DIR, side='b' if flip else 'w')
            st.success(f"Rozpoznany FEN: {fen}")
        else:
            fen = fen_input

        try:
            board = chess.Board(fen)
        except ValueError as e:
            st.error(f"Niepoprawny FEN: {e}")
            st.stop()

        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                result = engine.play(board, chess.engine.Limit(time=1))
                move = result.move

                st.info(f"Najlepszy ruch (notacja): {move}")

                # Tłumaczenie ruchu na opisowy
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
                st.success(f"Najlepszy ruch (opis): zagraj {piece_name} na pole {to_square}")

        except Exception as e:
            st.error(f"Błąd podczas działania silnika Stockfish: {e}")
