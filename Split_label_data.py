import os
import cv2
import shutil

# mapa figur FEN → etykieta
FEN_MAP = {
    'r': 'br', 'n': 'bn', 'b': 'bb', 'q': 'bq', 'k': 'bk', 'p': 'bp',
    'R': 'wr', 'N': 'wn', 'B': 'wb', 'Q': 'wq', 'K': 'wk', 'P': 'wp'
}

# funkcja: FEN → tablica 8x8 z etykietami
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
    return board  # 8 wierszy, od góry (rząd 8) do dołu (rząd 1)

# główna funkcja przetwarzająca obrazy i FEN
def process_chess_images(input_dir='resized_images', output_dir='dataset', target_base='chess5'):
    import os
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    fname = target_base + '.png'
    fen_path = os.path.join(input_dir, target_base + '.fen')
    img_path = os.path.join(input_dir, fname)

    if not os.path.exists(fen_path):
        print(f"Brak FEN dla {fname}, pomijam.")
        return

    with open(fen_path, 'r') as f:
        fen = f.read()

    labels = fen_to_labels(fen)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Błąd wczytywania {fname}")
        return

    h, w = img.shape[:2]
    tile_h = h // 8
    tile_w = w // 8

    for row in range(8):  # od góry do dołu
        for col in range(8):  # od lewej do prawej
            y1 = row * tile_h
            y2 = (row + 1) * tile_h
            x1 = col * tile_w
            x2 = (col + 1) * tile_w
            square_img = img[y1:y2, x1:x2]

            label = labels[row][col]
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            file_name = f"{target_base}_{chr(ord('a') + col)}{8 - row}.png"
            cv2.imwrite(os.path.join(label_dir, file_name), square_img)

    print(f"Obrobiono: {fname}")


process_chess_images()
