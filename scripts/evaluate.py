import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from datetime import datetime
from collections import defaultdict
from collections import Counter

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
else:
    print("GPU not available")

PROJ_DIR = str(Path(__file__).parents[1]) + "/"
print("project dir:" + PROJ_DIR)
DATA_DIR = PROJ_DIR + "dataset/"
print("data dir:" + DATA_DIR)
SAVE_DIR = PROJ_DIR + "models/"
print("save dir:" + SAVE_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_dataset = datasets.ImageFolder(DATA_DIR + "val", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

CLASS_NAMES = test_dataset.classes

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(torch.load(SAVE_DIR + "best_phase2.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


print("Konfidencja po klasie:")
conf_by_class = defaultdict(list)
for pred, true, conf in zip(all_preds, all_labels, all_confidences):
    conf_by_class[true].append(conf)
for c in conf_by_class:
    print(c, np.mean(conf_by_class[c]))

# REPORT
print(f"Dokładność ogólna: {accuracy_score(all_labels, all_preds):.4f}\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))
# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, title, fmt in [
    (axes[0], cm, "Liczby bezwzględne", "d"),
    (axes[1], cm_normalized, "Znormalizowana%", ".2f"),
]:
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Przewidziana klasa")
    ax.set_ylabel("Prawdziwa klasa")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.suptitle("Macierz pomyłek - klasyfikacja biomów z Minecraft'a", fontsize=14)
plt.tight_layout()
plt.savefig(SAVE_DIR + "confision_matrix.png", dpi=120)
plt.show()


def show_mistakes(model, dataset, device, class_names, max_images=12):
    """Wyświetla obrazy, które model sklasyfikował błędnie."""
    model.eval()
    mistakes = []

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for image, label in loader:
            output = model(image.to(device))
            pred = output.argmax(dim=1).item()
            true = label.item()

            if pred != true:
                img = image.squeeze().permute(1, 2, 0).numpy()
                img = img * IMAGENET_STD + IMAGENET_MEAN
                img = np.clip(img, 0, 1)
                mistakes.append((img, true, pred))

            if len(mistakes) >= max_images:
                break

    cols = 4
    rows = (len(mistakes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for ax, (img, true, pred) in zip(axes, mistakes):
        ax.imshow(img)
        ax.set_title(
            f"✓ {class_names[true]}\n✗ {class_names[pred]}", fontsize=9, color="red"
        )
        ax.axis("off")

    # Ukryj puste subploty
    for ax in axes[len(mistakes) :]:
        ax.axis("off")

    plt.suptitle("Błędnie sklasyfikowane obrazy", fontsize=13)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + "mistakes.png", dpi=120)
    plt.show()


print(Counter(all_labels))
print(len(test_dataset))
print(test_dataset.classes)
print("pred sample:", preds[:10])
print("labels sample:", labels[:10])
print("Mean confidence:", np.mean(all_confidences))
print("Min confidence:", np.min(all_confidences))
print("Max confidence:", np.max(all_confidences))

show_mistakes(model, test_dataset, DEVICE, CLASS_NAMES)

# REPORT
report = classification_report(
    all_labels, all_preds, target_names=CLASS_NAMES, digits=3
)

with open(SAVE_DIR + "evaluation_report.txt", "w") as f:
    f.write(f"Data ewaluacji: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Model: models/best_phase2.pth\n")
    f.write(f"Dokładność ogólna: {accuracy_score(all_labels, all_preds):.4f}\n\n")
    f.write(report)
print("Raport zapisany do models/evaluation_report.txt")
