from src.data import create_dataloaders, CLASS_NAMES


def main():

    print("Loading Galaxy10 DECaLS...\n")

    train_loader, test_loader = create_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get one batch
    batch = next(iter(train_loader))

    images = batch["pixel_values"]
    labels = batch["label"]

    print("\nBatch information:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)

    print("\nFirst 10 labels:")

    for label in labels[:10]:
        label_number = label.item()

        print(
            f"{label_number} -> {CLASS_NAMES[label_number]}"
        )


if __name__ == "__main__":
    main()