import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torchvision

# ===== 1. TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Adjust if grayscale
])

# ===== 2. DATASET LOADING =====
data_dir = "lung_tumor_dataset"
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
valid_data = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

classes = train_data.classes

# ===== 3. DEBUG PRINTS =====
print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Classes: {classes}")

# ===== 4. SHOW SAMPLE IMAGES =====
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

def show_sample_images(images, labels):
    grid_img = torchvision.utils.make_grid(images[:4], nrow=4)
    np_img = grid_img.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 3))
    plt.imshow(np_img)
    plt.title("Sample Batch - " + ", ".join([classes[label] for label in labels[:4]]))
    plt.axis("off")
    plt.show()

show_sample_images(images, labels)

# ===== 5. MODEL SETUP (ResNet50) =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 20 layers for fine-tuning
for param in list(model.parameters())[-20:]:
    param.requires_grad = True

# Replace final FC layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 4)  # 4 classes
)

model = model.to(device)
print("ResNet50 ready!")

# ===== 6. LOSS & OPTIMIZER =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===== 7. TRAINING FUNCTION =====
def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # ===== Validation =====
        model.eval()
        val_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()

        val_loss /= len(valid_loader)
        val_acc = val_correct / len(valid_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print("✅ Saved new best model")

# ===== 8. START TRAINING =====
train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ===== 9. LOAD BEST MODEL =====
model.load_state_dict(torch.load("best_resnet_model.pth"))
model.eval()

# ===== 10. EVALUATE ON TEST SET =====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ===== 11. REPORT =====
print("\n--- Classification Report on Test Set ---")
print(classification_report(all_labels, all_preds, target_names=classes))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(all_labels, all_preds))
