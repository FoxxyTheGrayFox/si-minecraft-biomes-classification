import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import json
import os
from pathlib import Path
import argparse

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

CLASS_NAMES = [
    "basalt_deltas",
    "cherry_grove",
    "crimson_forest",
    "end_highlands",
    "forest",
    "lush_cave",
    "ocean",
    "the_end",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

predict_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_model(model_path=DATA_DIR + "best_phase2.pth"):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model, top_k=3):
    """Top_k przewidywań z pewnością %."""
    image = Image.open(image_path).convert("RGB")
    tensor = predict_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze()

    top_probs, top_indices = probs.topk(top_k)
    results = [
        {
            "biom": CLASS_NAMES[idx.item()],
            "pewnosc": round(prob.item() * 100, 2),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]
    return results


# OUTPUT
BIOME_EMOJI = {
    "cherry_grove": "🌸",
    "forest": "🌲",
    "lush_cave": "🌿",
    "ocean": "🌊",
    "crimson_forest": "🍄",
    "basalt_deltas": "🌋",
    "the_end": "🌕",
    "end_highlands": "🌓",
}


def print_results(results, image_path):
    print(f"\nObraz: {image_path}")
    print("─" * 36)

    for i, r in enumerate(results):
        emoji = BIOME_EMOJI.get(r["biom"], "?")
        bar = "█" * int(r["pewnosc"] / 5)
        marker = " ← przewidywany" if i == 0 else ""
        print(f"{emoji} {r['biom']:<18} {r['pewnosc']:>6.2f}%  {bar}{marker}")

    print("─" * 36)

    if results[0]["pewnosc"] < 60:
        print("Niska pewność — obraz może być na granicy biomów")


# FOLDER PREDICTION
def predict_folder(folder_path, model, extensions=(".jpg", ".png", ".jpeg")):
    """Klasyfikuje wszystkie obrazy w folderze.
    Zapisuje wyniki do results.json
    """
    folder = Path(folder_path)
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"Brak obrazów w {folder_path}")
        return

    print(f"Znaleziono {len(images)} obrazów w {folder_path}\n")
    all_results = {}

    for img_path in sorted(images):
        results = predict(img_path, model)
        print_results(results, img_path.name)
        all_results[img_path.name] = results

    output_path = SAVE_DIR + "/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nWyniki zapisane do {output_path}")
    return all_results


# CLI OUTPUT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klasyfikacja biomów")
    parser.add_argument("input", help="Ścieżka do folderu z obrazkami")
    parser.add_argument(
        "--model", default=DATA_DIR + "best_phase2.pth", help="Ścieżka do wag modelu"
    )
    parser.add_argument(
        "--top", type=int, default=3, help="Ilość przewidywań do wyświetlenia"
    )
    args = parser.parse_args()

    model = load_model(args.model)
    path = Path(args.input)

    if path.is_dir():
        predict_folder(path, model)
    elif path.is_file():
        results = predict(path, model, top_k=args.top)
        print_results(results, path.name)
    else:
        print(f"{path} nie istnieje")
