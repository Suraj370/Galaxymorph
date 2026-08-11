import torch

from src.data import create_dataloaders
from src.model import GalaxyCNN
from src.train import train_model


SEED = 42


def main():

    # -----------------------------------------
    # Reproducibility
    # -----------------------------------------

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(
        f"Random seed: {SEED}"
    )

    print(
        "\nLoading Galaxy10 DECaLS...\n"
    )

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------

    (
        train_loader,
        test_loader,
        _,
    ) = create_dataloaders()

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    # -----------------------------------------
    # GPU
    # -----------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "PyTorch cannot use an NVIDIA GPU."
        )

    device = torch.device("cuda")

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Training device: {device}"
    )

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = GalaxyCNN()

    model = model.to(device)

    # -----------------------------------------
    # Train
    # -----------------------------------------

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=10,
        learning_rate=0.001,
        device=device,
    )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    main()