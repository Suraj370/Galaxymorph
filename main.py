import torch

from src.data import create_dataloaders
from src.model import GalaxyCNN


def main():

    print("Loading Galaxy10 DECaLS...")

    train_loader, test_loader = create_dataloaders()

    batch = next(iter(train_loader))

    images = batch["pixel_values"]
    labels = batch["label"]

    print("Input:", images.shape)
    print("Labels:", labels.shape)

    model = GalaxyCNN()

    output = model(images)

    print("Model output:", output.shape)


if __name__ == "__main__":
    main()