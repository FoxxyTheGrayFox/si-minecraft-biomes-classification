import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import datasets, transforms
from pathlib import Path

print(
    os.environ.get("HSA_OVERRIDE_GFX_VERSION")
)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.version.hip)
else:
    print("GPU not available")
PROJ_DIR = str(Path(__file__).parents[1]) + "/"
print("project dir:" + PROJ_DIR)
DATA_DIR = PROJ_DIR + "dataset/"
print("data dir:" + DATA_DIR)
SAVE_DIR = PROJ_DIR + "models/"
print("save dir:" + SAVE_DIR)

# TRANSFORMS AND DATALOADER
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

train_dataset = datasets.ImageFolder(DATA_DIR + "train", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR + "val", transform=val_transforms)
test_dataset = datasets.ImageFolder(DATA_DIR + "test", transform=val_transforms)

print("Klasy:", train_dataset.classes)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# MODEL EFFICIENTNET-B0
NUM_CLASSES = 8

def build_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

model = build_model(freeze_backbone=True)

total = sum(p.numel() for p in model.parameters())
active = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametry łącznie: {total:,}")
print(f"Parametry aktywne: {active:,}")

# OPTIMIZER
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używam: {DEVICE}")

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

# TRAIN AND EVALUATE
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

# BEST MODEL
def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path,
):
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            saved_marker = " zapisano"
        else:
            saved_marker = ""

        print(
            f"Epoka {epoch:>3}/{epochs} | "
            f"train_loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f} acc: {val_acc:.3f}"
            f"{saved_marker}"
        )
    return history

os.makedirs("models", exist_ok=True)

print("Trenowanie głowicy")
history_phase1 = run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    DEVICE,
    epochs=10,
    save_path=SAVE_DIR + "best_phase1.pth",  # zmieniłem
)

print("Fine-tuning")

for param in model.features.parameters():
    param.requires_grad = True

optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

history_phase2 = run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer_ft,
    scheduler_ft,
    DEVICE,
    epochs=20,
    save_path=SAVE_DIR + "best_phase2.pth",  # zmieniłem
)

def plot_history(h1, h2):
    train_loss = h1["train_loss"] + h2["train_loss"]
    val_loss = h1["val_loss"] + h2["val_loss"]
    train_acc = h1["train_acc"] + h2["train_acc"]
    val_acc = h1["val_acc"] + h2["val_acc"]
    epochs = list(range(1, len(train_loss) + 1))
    phase2_start = len(h1["train_loss"]) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, train, val, title in [
        (ax1, train_loss, val_loss, "Loss"),
        (ax2, train_acc, val_acc, "Accuracy"),
    ]:
        ax.plot(epochs, train, label="train")
        ax.plot(epochs, val, label="val")
        ax.axhline(
            phase2_start, color="gray", linestyle="--", label="fine-tuning start"
        )
        ax.set_title(title)
        ax.set_xlabel("Epoka")
        ax.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR + "training_history.png", dpi=120)
    plt.show()


plot_history(history_phase1, history_phase2)
