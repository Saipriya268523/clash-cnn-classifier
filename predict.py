"""
Predict clash / clean for a BIM screenshot and save an annotated image.

Usage:
    python predict.py --image view.png --weights models/clash_resnet18.pt
"""
import argparse
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["clash", "clean"]  # alphabetical order = ImageFolder order
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(weights):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    return model.to(DEVICE).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/clash_resnet18.pt")
    ap.add_argument("--out", default="results/prediction.png")
    args = ap.parse_args()

    model = load_model(args.weights)
    img = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(DEVICE)), 1)[0].cpu()

    p = dict(zip(CLASSES, probs.tolist()))
    label = max(p, key=p.get).upper()
    text = f"{label}  (clash {p['clash']:.1%} | clean {p['clean']:.1%})"
    print(text)

    bgr = cv2.imread(args.image)
    color = (0, 0, 255) if label == "CLASH" else (0, 150, 0)
    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(args.out, bgr)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
