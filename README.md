# Machine Learning Classification Project

This project implements and compares various machine learning methods for image classification, including traditional dimensionality reduction techniques and modern deep learning approaches.

## Project Overview

This is a project for an Introduction to Machine Learning course that focuses on:
- Image classification on 28×28 pixel images
- Dimensionality reduction using PCA
- Deep learning with MLPs, CNNs, and Vision Transformers
- Performance evaluation using accuracy and macro F1-score metrics

## Project Structure

```
├── main.py                    # Main entry point and training pipeline
├── dataset/                   # Dataset directory (train/test data)
├── src/
│   ├── data.py               # Data loading utilities
│   ├── utils.py              # Helper functions and evaluation metrics
│   └── methods/
│       ├── pca.py            # PCA dimensionality reduction
│       ├── deep_network.py   # Neural network implementations
│       └── dummy_methods.py  # Baseline dummy classifiers
└── report.pdf                # Project report
```

## Available Methods

### Dimensionality Reduction
- **PCA**: Principal Component Analysis for feature reduction

### Neural Networks
- **MLP**: Multi-Layer Perceptron with configurable hidden layers
- **CNN**: Convolutional Neural Network for image data
- **Vision Transformer**: Modern transformer-based architecture for images

### Baseline Methods
- **Dummy Classifier**: Random prediction baseline for comparison

## Installation & Requirements

The project requires Python with the following dependencies:
- `numpy`
- `torch` (PyTorch)
- `torchinfo`
- Additional ML libraries as needed

```bash
pip install torch torchvision numpy torchinfo
```

## Usage

### Basic Usage

Run the main script with default settings:
```bash
python main.py
```

### Advanced Configuration

The script supports various command-line arguments for customization:

```bash
python main.py \
    --data dataset \
    --nn_type mlp \
    --lr 0.001 \
    --max_iters 50 \
    --nn_batch_size 256 \
    --use_pca \
    --pca_d 100
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `dataset` | Path to dataset directory |
| `--nn_type` | `mlp` | Network architecture: `mlp`, `cnn`, or `transformer` |
| `--lr` | `0.001` | Learning rate for training |
| `--max_iters` | `10` | Number of training epochs |
| `--nn_batch_size` | `256` | Batch size for neural network training |
| `--device` | `cpu` | Training device: `cpu`, `cuda`, or `mps` |
| `--use_pca` | `False` | Enable PCA preprocessing |
| `--pca_d` | `100` | Number of principal components to keep |
| `--test` | `False` | Use test set for evaluation (otherwise uses validation split) |

## Dataset Format

The project expects the following files in the dataset directory:
- `train_data.npy`: Training images (N × H × W)
- `train_label.npy`: Training labels (N,)
- `test_data.npy`: Test images (N' × H × W)

Images are expected to be 28×28 pixels, suitable for MNIST-style datasets.

## Model Architectures

### MLP (Multi-Layer Perceptron)
- Fully connected layers: 512 → 128 → 64 → 32 → n_classes
- ReLU activation functions
- Input: Flattened image vectors

### CNN (Convolutional Neural Network)
- Convolutional layers with pooling
- Designed for 28×28 single-channel images
- Automatically handles spatial structure

### Vision Transformer
- Patch-based transformer architecture
- 7×7 patches, 2 transformer blocks
- 8 attention heads, 64-dimensional hidden layer

## Performance Metrics

The project evaluates models using:
- **Accuracy**: Percentage of correctly classified samples
- **Macro F1-Score**: Average F1-score across all classes (handles class imbalance)

## Training Pipeline

1. **Data Loading**: Load and reshape image data
2. **Preprocessing**: Normalize data and create train/validation split
3. **Dimensionality Reduction**: Optional PCA preprocessing
4. **Model Training**: Train selected neural network architecture
5. **Evaluation**: Report performance on training and validation/test sets

## Key Features

- **Modular Design**: Easy to add new methods and architectures
- **Flexible Configuration**: Extensive command-line options
- **Multiple Architectures**: Traditional ML and modern deep learning
- **Proper Evaluation**: Train/validation splits with multiple metrics
- **Performance Monitoring**: Training and inference time tracking