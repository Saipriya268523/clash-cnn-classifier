"""
Clash vs Clean classifier (BIM screenshots) - PyTorch transfer learning.

Folder layout expected before running:
    raw/clash/*.png
    raw/clean/*.png

Install:  pip install torch torchvision scikit-learn pillow
Run:      python clash_cnn.py
"""
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
RAW_DIR = Path("raw")
DATA_DIR = Path("dataset")
IMG_SIZE = 224
BATCH = 16
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------- 1. split
def split_dataset(train=0.7, val=0.15):
    """Split raw images into train/val/test BEFORE augmentation (no leakage)."""
    if DATA_DIR.exists():
        return
    for cls in ["clash", "clean"]:
        files = sorted(
            p for p in (RAW_DIR / cls).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        random.shuffle(files)
        n = len(files)
        n_tr, n_va = int(n * train), int(n * val)
        parts = {
            "train": files[:n_tr],
            "val": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:],
        }
        for split, items in parts.items():
            out = DATA_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy(f, out / f.name)
    print("Dataset split created in", DATA_DIR)


# ---------------------------------------------------------- 2. augmentation
# Applied on the fly: every epoch the model sees a different version of each image.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.0),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------- 3. model
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():          # freeze backbone first
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 2),
    )
    return model.to(DEVICE)


# ------------------------------------------------------------- 4. training
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    split_dataset()

    train_ds = datasets.ImageFolder(DATA_DIR / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", eval_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", eval_tf)
    print("Classes:", train_ds.class_to_idx)

    train_dl = DataLoader(train_ds, BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, BATCH)
    test_dl = DataLoader(test_ds, BATCH)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_val, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        # after 10 epochs, unfreeze last block and fine-tune with a small LR
        if epoch == 11:
            for p in model.layer4.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                [{"params": model.layer4.parameters(), "lr": 1e-5},
                 {"params": model.fc.parameters(), "lr": 1e-4}]
            )
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion)
        print(f"Epoch {epoch:02d} | train {tr_loss:.3f}/{tr_acc:.2%} "
              f"| val {va_loss:.3f}/{va_acc:.2%}")
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "clash_resnet18.pt")

    # ------------------------------------------------------- 5. final test
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            labels += y.tolist()
    print("\nTEST RESULTS")
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=test_ds.classes))


# ------------------------------------------------------------- 6. inference
def predict(image_path, weights="clash_resnet18.pt"):
    from PIL import Image
    model = build_model()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    img = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(img), 1)[0]
    classes = ["clash", "clean"]  # alphabetical order used by ImageFolder
    return {c: float(p) for c, p in zip(classes, prob)}


if __name__ == "__main__":
    main()
    # print(predict("some_new_screenshot.png"))
