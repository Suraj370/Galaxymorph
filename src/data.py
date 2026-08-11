from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    Image as HFImage,
    concatenate_datasets,
)
from torch.utils.data import DataLoader
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path(
    "datasets/matthieulel___galaxy10_decals/"
    "default/0.0.0/"
    "d1cfafdcce69ca7cd25aecce1a0b578cf07e6a30"
)

IMAGE_SIZE = 128
BATCH_SIZE = 32


# ============================================================
# Galaxy10 classes
# ============================================================

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


# ============================================================
# Load local dataset
# ============================================================

def load_local_dataset():

    train_datasets = []

    for i in range(5):

        file_path = (
            DATASET_PATH
            / f"galaxy10_decals-train-0000{i}-of-00005.arrow"
        )

        train_datasets.append(
            Dataset.from_file(str(file_path))
        )

    train_dataset = concatenate_datasets(train_datasets)

    test_path = DATASET_PATH / "galaxy10_decals-test.arrow"

    test_dataset = Dataset.from_file(str(test_path))

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    # Tell Hugging Face that the "image" column
    # contains image data and should be decoded.
    dataset = dataset.cast_column(
        "image",
        HFImage()
    )

    return dataset


# ============================================================
# Image transformations
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


# ============================================================
# Create DataLoaders
# ============================================================

def create_dataloaders():

    dataset = load_local_dataset()

    train_transform, test_transform = get_transforms()

    # --------------------------------------------------------
    # Training transformation
    # --------------------------------------------------------

    def transform_train(example):

        images = [
            train_transform(image)
            for image in example["image"]
        ]

        return {
            "pixel_values": images,
            "label": example["label"],
        }

    # --------------------------------------------------------
    # Testing transformation
    # --------------------------------------------------------

    def transform_test(example):

        images = [
            test_transform(image)
            for image in example["image"]
        ]

        return {
            "pixel_values": images,
            "label": example["label"],
        }

    # --------------------------------------------------------
    # Apply transformations
    # --------------------------------------------------------

    train_dataset = dataset["train"].with_transform(
        transform_train
    )

    test_dataset = dataset["test"].with_transform(
        transform_test
    )

    # --------------------------------------------------------
    # PyTorch DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset