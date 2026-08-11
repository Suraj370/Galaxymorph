from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.model import GalaxyCNN


CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
]


def load_best_model(device):
    """Load the best saved CNN checkpoint."""

    model = GalaxyCNN()

    model.load_state_dict(
        torch.load(
            "models/best_model.pth",
            map_location=device,
        )
    )

    model = model.to(device)

    return model


def plot_confusion_matrix(matrix):
    """Plot and save the confusion matrix."""

    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(matrix)

    plt.title("GalaxyMorph Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    # Add values to each cell
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = "results/confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to: {output_path}"
    )


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset."""

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    # -----------------------------
    # Calculate metrics
    # -----------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    # -----------------------------
    # Print overall results
    # -----------------------------

    print("\nTest Results")
    print("=" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # -----------------------------
    # Classification report
    # -----------------------------

    print("\nClassification Report")
    print("=" * 70)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print("\nConfusion Matrix")
    print("=" * 50)

    print(matrix)

    # -----------------------------
    # Save confusion matrix plot
    # -----------------------------

    plot_confusion_matrix(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }