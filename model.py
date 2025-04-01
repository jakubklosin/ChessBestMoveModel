import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Konfiguracja
BATCH_SIZE = 32
IMAGE_SIZE = 64
DATASET_PATH = 'dataset_split'
CLASSES = ['empty', 'wp', 'bp', 'wn', 'bn', 'wb', 'bb', 'wr', 'br', 'wq', 'bq', 'wk', 'bk']
NUM_CLASSES = len(CLASSES)

# Transformacje (w tym resize)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(f'{DATASET_PATH}/train', transform=transform)
val_dataset = datasets.ImageFolder(f'{DATASET_PATH}/val', transform=transform)
test_dataset = datasets.ImageFolder(f'{DATASET_PATH}/test', transform=transform)

# Loadery
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model z BatchNorm
class ChessSquareCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32 -> 16x16
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Trening
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChessSquareCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Ewaluacja na walidacji
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Accuracy: {acc:.2f}%")

# Zapis modelu
torch.save(model.state_dict(), "chess_cnn_model.pth")
