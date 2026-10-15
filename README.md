# BIM Clash Detection with a CNN (Image Classification)

A deep-learning project that classifies BIM (Revit/Navisworks-style) screenshots of MEP duct and structural models as **clash** or **clean**, using transfer learning with ResNet18 in PyTorch.

> Companion project: object-detection version with bounding boxes using YOLOv8 -> **[add your YOLOv8 repo link here]**

## Example result

![Example prediction](results/example_prediction.png)

The model flags this view as a clash with 80.9% confidence.

## Approach

1. **Data:** screenshots of clash and clean views collected from BIM models (~60 per class).
2. **Split first, augment second:** train/val/test split (70/15/15) is made *before* augmentation so augmented copies never leak into validation or test.
3. **Augmentation (training set only):** random resized crop, horizontal flip, small rotation, brightness/contrast jitter, light Gaussian blur.
4. **Model:** ImageNet-pretrained ResNet18. The backbone is frozen at first, then the last block (`layer4`) is fine-tuned with a low learning rate.
5. **Localisation (optional):** Grad-CAM + OpenCV contours turn the model's attention map into approximate bounding boxes (`notebooks/03_gradcam_boxes.ipynb`).

## Results

| Metric (test set) | Value |
|---|---|
| Accuracy | 99.4% |
| Clash recall | 99.4% |
| Clash precision | 99.4% |

*(Replace with your own numbers from the confusion matrix / classification report.)*

## Repository structure

```
.
├── notebooks/
│   ├── 01_augment_dataset.ipynb   # upload zip, split, augment (Colab)
│   ├── 02_train_cnn.ipynb         # train + evaluate ResNet18 (Colab)
│   └── 03_gradcam_boxes.ipynb     # Grad-CAM + OpenCV boxes (Colab)
├── train.py                       # local training script
├── predict.py                     # inference on a single screenshot
├── models/                        # trained weights (clash_resnet18.pt)
├── results/                       # example outputs
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt

# Train (expects raw/clash/*.png and raw/clean/*.png)
python train.py

# Predict on a new screenshot
python predict.py --image my_view.png --weights models/clash_resnet18.pt
```

Or run the notebooks in order on Google Colab (GPU recommended).

## Limitations

- Small dataset (~120 original images); augmentation adds variety but not new scenes, so the model may not generalise to unseen projects.
- Classification only says *whether* a clash is visible, not its exact location. Grad-CAM boxes are approximate.
- Geometry-based tools (Navisworks Clash Detective, Revit Interference Check) remain the exact method. This project explores ML as a triage aid for clash screenshots.

## Future work

- Collect more real screenshots across projects and disciplines.
- Multi-class clash types (duct-beam, duct-pipe, duct-column).
- Evaluate on a completely unseen project.
- Deploy as a Gradio/Streamlit demo.

## Tech stack

Python, PyTorch, torchvision, OpenCV, scikit-learn, Google Colab
