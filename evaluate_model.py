import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_def import ChessSquareCNN
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Klasy
CLASSES = ['empty', 'wp', 'bp', 'wn', 'bn', 'wb', 'bb', 'wr', 'br', 'wq', 'bq', 'wk', 'bk']
NUM_CLASSES = len(CLASSES)

# Ustawienia
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dane
eval_dataset = datasets.ImageFolder('eval_pola', transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Model
model = ChessSquareCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('chess_cnn_model.pth', map_location=device))
model.eval()

# Predykcje
results = []
y_true = []
y_pred = []

with torch.no_grad():
    for img, label in eval_loader:
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        pred = torch.argmax(output, 1)

        true_class = CLASSES[label.item()]
        pred_class = CLASSES[pred.item()]

        results.append({
            "Plik": os.path.basename(eval_dataset.samples[len(y_true)][0]),
            "Prawdziwa etykieta": true_class,
            "Predykcja": pred_class,
            "Poprawna?": true_class == pred_class
        })

        y_true.append(label.item())
        y_pred.append(pred.item())

# Tabela wyników
df = pd.DataFrame(results)
print(df)
print(f"\nDokładność: {(df['Poprawna?'].sum() / len(df)) * 100:.2f}%")

labels_in_eval = sorted(set(y_true + y_pred))
class_names_in_eval = [CLASSES[i] for i in labels_in_eval]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_in_eval)
disp.plot(xticks_rotation=45)
plt.title("Macierz pomyłek – eval_pola")
plt.tight_layout()
plt.show()
