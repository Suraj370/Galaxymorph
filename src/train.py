from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


MODEL_PATH = Path("models/best_model.pth")


def train_one_epoch(
    model,
    train_loader,
    loss_function,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        # Move batch to GPU
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass on GPU
        outputs = model(images)

        # Loss on GPU
        loss = loss_function(outputs, labels)

        # Backpropagation on GPU
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return average_loss, accuracy


def validate(
    model,
    test_loader,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in test_loader:

            # Move validation batch to GPU
            images = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = loss_function(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return average_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    learning_rate=0.001,
    device=None,
):

    # -----------------------------------------
    # Select GPU
    # -----------------------------------------

    if device is None:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available. "
                "Install a CUDA-enabled PyTorch build "
                "or use a machine with an NVIDIA GPU."
            )

        device = torch.device("cuda")

    print(f"Training device: {device}")

    # Move entire model to GPU
    model = model.to(device)

    # Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = validate(
            model=model,
            test_loader=test_loader,
            loss_function=loss_function,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}]"
        )

        print(
            f"  Train Loss:      {train_loss:.4f}"
        )

        print(
            f"  Train Accuracy:  {train_accuracy:.4f}"
        )

        print(
            f"  Val Loss:        {val_loss:.4f}"
        )

        print(
            f"  Val Accuracy:    {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                f"  ✓ Saved best model "
                f"({best_accuracy:.4f})"
            )

        print()

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Model saved at: {MODEL_PATH}"
    )

    return model, history