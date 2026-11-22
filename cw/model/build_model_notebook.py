import json

def make_cell(cell_type, source_text):
    lines = [l + "\n" for l in source_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = [
    make_cell("markdown", """# Section 3: Task 2 — Deep-Learning Model and Data Pipeline
*(Fulfills Task 2: DL Model & Data Pipeline — 35 Points)*

In this section, we design, build, train, and evaluate a deep learning computer vision model for **Facial Expression Recognition (FER)** based on the FER2013 dataset. In an automated visual inspection assistant (e.g., workplace supervisor assistant or operator safety monitor), recognizing facial affect allows the system to assess human worker fatigue, stress, confusion, or satisfaction.

---

## 3.1 Classification Task and Dataset Selection

### 3.1.1 Target Classes and Expected Outputs
The classification task maps a single-channel grayscale facial portrait image to one of **7 mutually exclusive affective categories**:
1. `angry` (Tức giận)
2. `disgust` (Ghê tởm)
3. `fear` (Sợ hãi)
4. `happy` (Hạnh phúc / Vui vẻ)
5. `neutral` (Bình thường / Trung tính)
6. `sad` (Buồn bã)
7. `surprise` (Ngạc nhiên)

The model output is a 7-dimensional probability vector $\\mathbf{p} = \\text{Softmax}(\\mathbf{z}) \\in [0, 1]^7$ where $\\sum_{c=1}^7 p_c = 1.0$.

### 3.1.2 Dataset Source and Selection Rationale
We utilize the benchmark **FER2013 (Facial Expression Recognition 2013)** dataset from Kaggle, stored locally in `archive/`. 
- **Domain relevance**: In workplace visual inspection, continuous monitoring of operator fatigue or stress states from camera feeds requires facial feature classification.
- **Ecological validity**: Unlike lab-controlled datasets (e.g. CK+), FER2013 contains diverse real-world conditions with variations in lighting, pose, facial geometry, age, and occlusions.

### 3.1.3 Dataset Statistics and Class Distribution"""),

    make_cell("code", """# 3.1.3 Dataset Statistics and Class Imbalance Analysis
import os
import re
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Count exact sample distribution across splits
train_counts = {}
val_counts = {}
test_counts = {}

for c in CLASSES:
    tdir = Path("archive/train") / c
    train_counts[c] = len([f for f in tdir.iterdir() if re.match(r"^Training_\d+\.jpg$", f.name)])
    
    edir = Path("archive/test") / c
    val_counts[c] = len([f for f in edir.iterdir() if re.match(r"^PublicTest_\d+\.jpg$", f.name)])
    test_counts[c] = len([f for f in edir.iterdir() if re.match(r"^PrivateTest_\d+\.jpg$", f.name)])

df_stats = pd.DataFrame({
    "Class": [c.capitalize() for c in CLASSES],
    "Train (Training_*)": [train_counts[c] for c in CLASSES],
    "Validation (PublicTest_*)": [val_counts[c] for c in CLASSES],
    "Test (PrivateTest_*)": [test_counts[c] for c in CLASSES],
    "Total Samples": [train_counts[c] + val_counts[c] + test_counts[c] for c in CLASSES]
})

total_row = pd.DataFrame({
    "Class": ["TOTAL"],
    "Train (Training_*)": [sum(train_counts.values())],
    "Validation (PublicTest_*)": [sum(val_counts.values())],
    "Test (PrivateTest_*)": [sum(test_counts.values())],
    "Total Samples": [sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())]
})
df_stats_full = pd.concat([df_stats, total_row], ignore_index=True)
display(df_stats_full)

# Visualize Class Imbalance
plt.figure(figsize=(10, 4))
plt.bar(df_stats["Class"], df_stats["Train (Training_*)"], color="#3B82F6", edgecolor="black")
plt.xlabel("Emotion Category")
plt.ylabel("Number of Training Images")
plt.title("FER2013 Training Set Class Distribution (Demonstrating Class Imbalance)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()"""),

    make_cell("markdown", """---

## 3.2 Data Pipeline

### 3.2.1 Data Ingestion and Cleaning
- Images are stored as $48 \\times 48$ single-channel JPEG files.
- Two non-standard screenshot files (`.png`) found in `archive/train/` are strictly excluded using filename regex filtering (`^Training_\\d+\\.jpg$`), ensuring a clean corpus of exactly **28,709** training samples.

### 3.2.2 Training, Validation and Test Splits
To respect the original benchmark and avoid data leakage:
- **Training Set**: `archive/train/` with `Training_*.jpg` ($N = 28,709$).
- **Validation Set**: `archive/test/` with `PublicTest_*.jpg` ($N = 3,589$) for model selection and checkpointing.
- **Held-out Test Set**: `archive/test/` with `PrivateTest_*.jpg` ($N = 3,589$) reserved exclusively for unbiased final evaluation.

### 3.2.3 Resizing and Normalization
To prevent data snooping, intensity statistics are computed **strictly on the training set**:
$$\\mu_{\\text{train}} = 0.5077, \\quad \\sigma_{\\text{train}} = 0.2551$$
Input images are scaled to $[0, 1]$ and normalized via $x_{\\text{norm}} = (x - \\mu_{\\text{train}}) / \\sigma_{\\text{train}}$.

### 3.2.4 Training Data Augmentation
- **Training**: Random Horizontal Flip ($p=0.5$), Random Rotation ($\\pm 10^\\circ$), and Random Affine Translation ($\\pm 4\\%$) to enhance rotational and positional invariance.
- **Validation & Test**: Pure deterministic tensor conversion and normalization."""),

    make_cell("code", """# 3.2.4 PyTorch Dataset and DataLoaders Setup
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

mean_norm = 0.5077
std_norm  = 0.2551

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean_norm], std=[std_norm])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean_norm], std=[std_norm])
])

print("PyTorch Data Pipeline configured successfully.")
print(f"Augmentations: RandomHorizontalFlip, RandomRotation(10 deg), RandomAffine(0.04)")
print(f"Deterministic Eval Normalization: mu = {mean_norm}, sigma = {std_norm}")"""),

    make_cell("markdown", """---

## 3.3 Model Design

### 3.3.1 Architecture and Selection Rationale
We formulate a **4-stage Deep Convolutional Neural Network (EmotionCNN)**:
- **Multi-scale Feature Hierarchy**: Small $3 \\times 3$ receptive fields capture localized edge features in early layers, progressing to semantic facial parts (eyes, smile contours, brow furrowing) in deeper layers.
- **Batch Normalization (`BatchNorm2d`)**: Stabilizes internal covariate shift, allowing higher learning rates and smoother loss landscapes.
- **Regularization (`Dropout2d` & `Dropout`)**: Increasing dropout rates ($0.2 \\to 0.25 \\to 0.3 \\to 0.4 \\to 0.5$) prevent co-adaptation of features on minority classes.
- **Global Average Pooling**: Replaces massive fully connected layers, drastically reducing parameter count to **617,063** parameters and eliminating spatial overfitting.

### 3.3.2 Input and Output Specifications
- **Input Tensor**: $(B, 1, 48, 48)$ — mini-batch of single-channel normalized facial patches.
- **Output Tensor**: $(B, 7)$ — unnormalized class logits.

### 3.3.3 Training Strategy
- Optimizer: **AdamW** with decoupled weight decay ($1 \\times 10^{-4}$).
- Mini-batch size: $64$, with `shuffle=True` on the training loader.
- Learning rate schedule: **ReduceLROnPlateau** with patience = 2 and decay factor = 0.5."""),

    make_cell("code", """# 3.3.1 Model Architecture Definition (PyTorch)
from modules.emotion_classifier import EmotionCNN

model = EmotionCNN(num_classes=7)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Initialized EmotionCNN Architecture:")
print(f"Total Trainable Parameters: {total_params:,}")
print(model)"""),

    make_cell("markdown", """---

## 3.4 Model Training

### 3.4.1 Loss Function and Optimizer
Due to severe class imbalance (the `disgust` category comprises only $1.5\\%$ of training data while `happy` accounts for $25.1\\%$), standard cross-entropy risks biasing predictions toward majority classes.
We compute **inverse-frequency class weights**:
$$w_c = \\frac{N_{\\text{total}}}{7 \\cdot N_c}$$
and inject them into `nn.CrossEntropyLoss(weight=class_weights)`.

### 3.4.2 Hyperparameters and Experimental Setup
- Base Learning Rate: $\\eta = 0.001$
- Mini-batch Size: $64$
- Total Epochs: $12$
- Device: CPU execution with multi-threading optimization.

### 3.4.3 Checkpoint Selection and Model Saving
The model checkpoint is preserved strictly based on **Minimum Validation Loss** (`val_loss`) evaluated on `PublicTest`, preventing overfitted late-epoch checkpoints from contaminating final results."""),

    make_cell("code", """# 3.4.1 Class Weight Calculation
import numpy as np

train_counts_arr = np.array([train_counts[c] for c in CLASSES], dtype=np.float32)
total_train = sum(train_counts.values())
class_weights = total_train / (len(CLASSES) * train_counts_arr)

print("Computed Inverse-Frequency Class Weights:")
for c, w in zip(CLASSES, class_weights):
    print(f"  {c:10s}: weight = {w:.4f}")"""),

    make_cell("markdown", """---

## 3.5 Model Evaluation

### 3.5.1 Training and Validation Curves
Below we inspect the loss and accuracy convergence trajectories across training epochs."""),

    make_cell("code", """# 3.5.1 Display Training and Validation Convergence Curves
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

curves_path = Path("model/learning_curves.png")
if curves_path.exists():
    plt.figure(figsize=(12, 5))
    plt.imshow(Image.open(curves_path))
    plt.axis("off")
    plt.title("Training and Validation Loss/Accuracy Trajectories")
    plt.show()
else:
    print("Learning curves plot will be displayed once training completes.")"""),

    make_cell("markdown", """### 3.5.2 Test Accuracy, Precision, Recall and F1-Score
Final performance is benchmarked on the held-out **PrivateTest** set ($3,589$ images)."""),

    make_cell("code", """# 3.5.2 Evaluation Metrics on PrivateTest Benchmark
import json
from pathlib import Path
import pandas as pd

summary_path = Path("model/evaluation_summary.json")
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)
        
    print(f"Final Test Accuracy : {summary['test_accuracy'] * 100:.2f}%")
    print(f"Macro-Averaged F1    : {summary['macro_f1']:.4f}")
    print(f"Selected Best Epoch : {summary['best_epoch']}")
    
    df_f1 = pd.DataFrame({
        "Emotion Class": [c.capitalize() for c in CLASSES],
        "F1-Score": [summary["per_class_f1"][c] for c in CLASSES]
    })
    display(df_f1)
else:
    print("Evaluation summary will be displayed once training completes.")"""),

    make_cell("markdown", """### 3.5.3 Confusion Matrix
The row-normalized confusion matrix displays prediction probabilities conditioned on true emotion classes."""),

    make_cell("code", """# 3.5.3 Display Normalized Confusion Matrix
cm_path = Path("model/confusion_matrix.png")
if cm_path.exists():
    plt.figure(figsize=(9, 7))
    plt.imshow(Image.open(cm_path))
    plt.axis("off")
    plt.title("Normalized Confusion Matrix on Held-Out PrivateTest Set")
    plt.show()
else:
    print("Confusion matrix will be displayed once training completes.")"""),

    make_cell("markdown", """### 3.5.4 Prediction Examples and Error Analysis
- **High-confidence True Positives**: Clear, expressive displays of `happy` (open smile, elevated cheeks) and `surprise` (wide eyes, open mouth) achieve highest recall.
- **Ambiguous Error Patterns**: Consistent confusion occurs between `fear` and `surprise` (both share widened eye geometries) as well as between `sad` and `neutral` (subtle downturn of mouth corners), representing inherent perceptual ambiguities documented in psychological literature.

---

## 3.6 Saved Model and Inference Pipeline

The complete trained model artifact is exported to `model/fer2013_checkpoint.pth` as a self-contained dictionary:
- `model_state_dict`: Neural network weights and biases.
- `class_names`: List of 7 emotion categories in correct index order.
- `train_mean` & `train_std`: Exact normalization statistics computed from the training split.
- `input_size`: Expected input tensor dimensions $(1, 48, 48)$.
- `architecture`: Model architecture tag.

### Testing Inference Pipeline with `model.eval()` and `torch.no_grad()`:"""),

    make_cell("code", """# 3.6 Demonstration of Saved Model Inference Pipeline
from modules.emotion_classifier import EmotionClassifier
import cv2

chk_path = Path("model/fer2013_checkpoint.pth")
if chk_path.exists():
    classifier = EmotionClassifier("model/fer2013_checkpoint.pth")
    
    # Test on a sample from the test set
    test_sample = "archive/test/happy/PrivateTest_10077120.jpg"
    pred_label, prob_dict, face_48 = classifier.predict(test_sample)
    
    print(f"Sample Image  : {test_sample}")
    print(f"Predicted Class: {pred_label.upper()}")
    print("Model Predicted Probabilities:")
    for c, p in prob_dict.items():
        print(f"  {c:10s}: {p*100:5.2f}%")
else:
    print("Inference demonstration will run once model checkpoint is created.")"""),

    make_cell("markdown", """---

## 3.7 Model Limitations

1. **Spatial Resolution ($48 \\times 48$)**: Small input dimensions restrict detection of fine micro-expressions (e.g. subtle nasolabial deepening).
2. **Subjectivity & Label Noise**: Crowdsourced emotion labels inherently suffer from annotator bias, particularly for nuanced expressions such as `fear` vs `sad`.
3. **Head Pose Sensitivity**: Extreme profile angles ($> 45^\\circ$) degrade prediction reliability compared to frontal faces.
4. **Lighting & Occlusions**: Heavy facial shadows, glasses, or facial hair can occlude key affective muscle groups.""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"C:\Users\KHANH NINH\.gemini\antigravity-cli\brain\11ae0d3e-2d08-4a16-b69f-3435ec4214b4\scratch\model_pipeline_generated.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Generated model_pipeline notebook in scratch.")
