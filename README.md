# GalaxyMorph

CNN-based galaxy morphology classification using the Galaxy10 DECaLS dataset and PyTorch.

## Project Goal

GalaxyMorph classifies galaxy images into 10 different morphological classes using a convolutional neural network (CNN).

## Dataset

Galaxy10 DECaLS.

- Training images: 15,962
- Test images: 1,774
- Image size: 256 × 256
- Model input size: 128 × 128
- Number of classes: 10

## Tech Stack

- Python
- PyTorch
- Torchvision
- Hugging Face Datasets
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure

```text
GalaxyMorph/
├── datasets/
├── models/
│   └── best_model.pth
├── src/
│   ├── data.py
│   ├── model.py
│   └── train.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md

```

## Model

The current model is a custom CNN consisting of:

- 3 convolutional blocks
- ReLU activations
- Max pooling
- Fully connected classifier
- Dropout

### Input

```text
3 × 128 × 128
```
## Output
```text
10 classes
```
## Training

Training is performed using:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 10
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

## Results

Initial CNN training:

| Metric | Result |
|---|---:|
| Best validation accuracy | 72.32% |
| Best epoch | 10 |
| Training accuracy | 69.15% |
| Validation loss | 0.8009 |

The best model is saved to:

```text
models/best_model.pth

```
## Running the Project
```bash
Install dependencies:


uv sync

Run training:


uv run main.py

```