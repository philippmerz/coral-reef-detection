import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, ViTForImageClassification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_classification_report(y_true, y_pred, labels=["CORAL", "CORAL_BL"]):
    """Plots the classification report as a table image."""
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    plt.figure(figsize=(8, len(df)*0.6+1))
    plt.title('Classification Report')
    sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt='.2f', cmap='YlGnBu', cbar=False)
    plt.yticks(rotation=0)
    plt.show()

def print_evaluation_matrix(y_true, y_pred, labels=["CORAL", "CORAL_BL"]):
    """Prints the confusion matrix for the predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (rows: true, cols: pred):")
    print(f"Labels: {labels}")
    print(cm)

def plot_evaluation_matrix(y_true, y_pred, labels=["CORAL", "CORAL_BL"]):
    """Plots the confusion matrix as an image."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def load_model():
    model_name = "akridge/noaa-esd-coral-bleaching-vit-classifier-v1"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

def majority_label(mask_bleached, mask_non_bleached):
    """Assign the label with more white pixels."""
    mask_b = np.array(mask_bleached.convert("L")) > 127
    mask_nb = np.array(mask_non_bleached.convert("L")) > 127

    frac_b = mask_b.mean()
    frac_nb = mask_nb.mean()

    if frac_b >= frac_nb:
        return "CORAL_BL"
    else:
        return "CORAL"


def evaluate_images(dataset_root, processor, model, device=torch.device("cpu")):
    images_dir = Path(dataset_root) / "images"
    masks_b_dir = Path(dataset_root) / "masks_bleached"
    masks_nb_dir = Path(dataset_root) / "masks_non_bleached"

    y_true, y_pred = [], []

    for img_file in images_dir.glob("*.jpg"):
        base = img_file.stem  # e.g. "image_1"

        mask_b_file = masks_b_dir / f"{base}_bleached.png"
        mask_nb_file = masks_nb_dir / f"{base}_non_bleached.png"

        if not mask_b_file.exists() or not mask_nb_file.exists():
            print(f"Missing masks for {base}")
            continue

        mask_b = Image.open(mask_b_file)
        mask_nb = Image.open(mask_nb_file)
        true_label = majority_label(mask_b, mask_nb)
        if true_label is None:
            print(f"Skipping {base} (ambiguous)")
            continue

        # Run pretrained model
        image = Image.open(img_file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = outputs.logits.argmax(-1).item()
            pred_label = model.config.id2label[pred_id]

        y_true.append(true_label)
        y_pred.append(pred_label)

        # print(f"{img_file.name}: true={true_label}, pred={pred_label}")

    print("\nSamples evaluated:", len(y_true))
    if y_true:
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Classification Report:")
        print(classification_report(y_true, y_pred, labels=["CORAL", "CORAL_BL"]))
        # Plot confusion matrix as image
        plot_evaluation_matrix(y_true, y_pred, labels=["CORAL", "CORAL_BL"])
        # Plot classification report as image
        plot_classification_report(y_true, y_pred, labels=["CORAL", "CORAL_BL"])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model()
    model = model.to(device)

    dataset_path = "C:/Users/20235050/Downloads/BDS_Y3/DC3/coral_bleaching/reef_support"
    evaluate_images(dataset_path, processor, model, device=device)

