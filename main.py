import torch

from src.data import create_dataloaders
from src.model import GalaxyCNN
from src.train import train_model


def main():

    print("Loading Galaxy10 DECaLS...\n")

    train_loader, test_loader = create_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # -----------------------------------------
    # Check GPU
    # -----------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "PyTorch cannot use an NVIDIA GPU."
        )

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = GalaxyCNN()

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

    print("\nTraining complete.")


if __name__ == "__main__":
    main()