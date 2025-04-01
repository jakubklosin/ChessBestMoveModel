# Środowisko zostało zresetowane — ponownie uruchamiam kompletny czysty kod

import torch
from torchvision import transforms
from PIL import Image
import os

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
                    predictions[field_code] = CLASSES[pred_idx]
    return predictions

# Konwersja predykcji do FEN
def predictions_to_fen(predictions):
    fen_rows = []
    for rank in range(8, 0, -1):
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
    return "/".join(fen_rows) + " w - - 0 1"

# Wykonanie
predictions = get_predictions()

def verify_predictions(predictions):
    all_squares = [f+str(r) for r in range(1, 9) for f in "abcdefgh"]
    missing = [sq for sq in all_squares if sq not in predictions]
    if missing:
        print(f"Brakujące pola: {missing}")
    return len(missing) == 0

for square in sorted(predictions):
    print(f"{square}: {predictions[square]}")
generated_fen = predictions_to_fen(predictions)
generated_fen
