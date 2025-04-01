import cv2
import os
import shutil

# mapa FEN → etykieta
FEN_MAP = {
    'r': 'br', 'n': 'bn', 'b': 'bb', 'q': 'bq', 'k': 'bk', 'p': 'bp',
    'R': 'wr', 'N': 'wn', 'B': 'wb', 'Q': 'wq', 'K': 'wk', 'P': 'wp'
}
if os.path.exists("eval_pola"):
    shutil.rmtree("eval_pola")

def fen_to_labels(fen_string):
    board = []
    rows = fen_string.strip().split()[0].split('/')
    for row in rows:
        row_data = []
        for ch in row:
            if ch.isdigit():
                row_data.extend(['empty'] * int(ch))
            else:
                row_data.append(FEN_MAP.get(ch, 'empty'))
        board.append(row_data)
    return board  # 8x8 etykiet

def process_eval_board(img_name='new_chess10', input_dir='resized_images', output_dir='eval_pola'):
    os.makedirs(output_dir, exist_ok=True)

    fen_path = os.path.join(input_dir, img_name + '.fen')
    img_path = os.path.join(input_dir, img_name + '.png')

    if not os.path.exists(fen_path):
        print(f"Brak pliku FEN: {fen_path}")
        return

    with open(fen_path, 'r') as f:
        fen = f.read()

    labels = fen_to_labels(fen)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Błąd wczytywania obrazu: {img_path}")
        return

    h, w = img.shape[:2]
    tile_h = h // 8
    tile_w = w // 8

    for row in range(8):
        for col in range(8):
            y1 = row * tile_h
            y2 = (row + 1) * tile_h
            x1 = col * tile_w
            x2 = (col + 1) * tile_w
            square_img = img[y1:y2, x1:x2]

            label = labels[row][col]
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            field_name = f"{img_name}_{chr(ord('a') + col)}{8 - row}.png"
            cv2.imwrite(os.path.join(label_dir, field_name), square_img)

    print(f"Obraz {img_name} przetworzony do katalogu: {output_dir}")

process_eval_board()
