import torch
import torch.nn as nn


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        images = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in data_loader:

            images = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs=10,
    learning_rate=0.001,
):
    """
    Train the baseline CNN using standard
    CrossEntropyLoss.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # -----------------------------------------
    # Standard CrossEntropyLoss
    # -----------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # -----------------------------------------
    # Best model tracking
    # -----------------------------------------

    best_accuracy = 0.0

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]"
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

        # -------------------------------------
        # Save best model based ONLY on
        # validation accuracy
        # -------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "models/best_model.pth",
            )

            print(
                f"  ✓ Saved best model "
                f"({best_accuracy:.4f})"
            )

    print(
        f"\nBest validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        "Model saved at: "
        "models/best_model.pth"
    )

    return model, history