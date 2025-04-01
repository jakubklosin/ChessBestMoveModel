import torch
from torchvision import transforms
from PIL import Image
import os
import json
import cv2
import shutil

# Wczytaj zapisane mapowanie klas
with open("class_index.json", "r") as f:
    class_to_idx = json.load(f)

# Odwróć mapowanie: indeks → nazwa klasy
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Mapowanie klas do symboli FEN
CLASS_TO_FEN = {
    'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R', 'wq': 'Q', 'wk': 'K',
    'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k',
    'empty': '1'
}
CLASSES = list(CLASS_TO_FEN.keys())
NUM_CLASSES = len(CLASSES)

# Model CNN
class ChessSquareCNN(torch.nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Przygotowanie modelu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChessSquareCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("chess_cnn_model.pth", map_location=device))
model.eval()

# Transformacja wejściowa
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Predykcja pól z eval_pola
def get_predictions(eval_dir="eval_pola"):
    predictions = {}
    for class_dir in os.listdir(eval_dir):
        class_path = os.path.join(eval_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if not fname.endswith('.png'):
                continue
            # Ekstrahuj kod pola z nazwy pliku w formacie "new_chess10_g8.png"
            parts = fname.split("_")
            if len(parts) >= 3:
                field_code = parts[-1].replace(".png", "")  # g8
                img_path = os.path.join(class_path, fname)
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    predictions[field_code] = idx_to_class[pred_idx]
    return predictions

# Konwersja predykcji do FEN
def predictions_to_fen(predictions, side='w'):
    fen_rows = []
    ranks = range(8, 0, -1) if side == 'w' else range(1, 9)
    for rank in ranks:
        row = ""
        empty_count = 0
        for file in "abcdefgh":
            square = f"{file}{rank}"
            label = predictions.get(square, 'empty')
            symbol = CLASS_TO_FEN[label]
            if symbol == '1':
                empty_count += 1
            else:
                if empty_count:
                    row += str(empty_count)
                    empty_count = 0
                row += symbol
        if empty_count:
            row += str(empty_count)
        fen_rows.append(row)
    if side == 'b':
        fen_rows = fen_rows[::-1]
    return "/".join(fen_rows) + f" {side} - - 0 1"

# Wykonanie
predictions = get_predictions()
generated_fen = predictions_to_fen(predictions)

def get_fen_from_eval_pola(eval_dir="eval_pola", side='w'):
    predictions = get_predictions(eval_dir)
    return predictions_to_fen(predictions, side=side)


FEN_MAP = {
    'r': 'br', 'n': 'bn', 'b': 'bb', 'q': 'bq', 'k': 'bk', 'p': 'bp',
    'R': 'wr', 'N': 'wn', 'B': 'wb', 'Q': 'wq', 'K': 'wk', 'P': 'wp'
}

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
    return board

def process_board_image(img_name, input_dir='resized_images', output_dir='eval_pola', flip=False):
    import shutil
    import cv2

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(input_dir, img_name + '.png')
    img = cv2.imread(img_path)
    if img is None:
        print(f"Błąd wczytywania obrazu: {img_path}")
        return

    if flip:
        img = cv2.flip(img, -1)  # obrót całej planszy o 180°

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

            if flip:
                square_img = cv2.rotate(square_img, cv2.ROTATE_180)

            label_dir = os.path.join(output_dir, 'unknown')
            os.makedirs(label_dir, exist_ok=True)

            field_name = f"{img_name}_{chr(ord('a') + col)}{8 - row}.png"
            cv2.imwrite(os.path.join(label_dir, field_name), square_img)

    print(f"Obraz {img_name} podzielony na pola w katalogu: {output_dir}")



if __name__ == "__main__":
    fen = get_fen_from_eval_pola()
    print("Wygenerowany FEN:", fen)

