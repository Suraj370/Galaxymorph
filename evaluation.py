import torch

from src.data import create_dataloaders
from src.evaluate import evaluate_model, load_best_model


def main():

    print("Loading Galaxy10 DECaLS...\n")

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------

    (
        _,
        _,
        test_loader,
    ) = create_dataloaders()

    print(
        f"Test batches: {len(test_loader)}"
    )

    # -----------------------------------------
    # GPU
    # -----------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    # -----------------------------------------
    # Load best model
    # -----------------------------------------

    print("\nLoading best model...")

    model = load_best_model(device)

    print(
        "Loaded: models/best_model.pth"
    )

    # -----------------------------------------
    # Evaluate on REAL test set
    # -----------------------------------------

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
    )


if __name__ == "__main__":
    main()