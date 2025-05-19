# CIFAR-10 Image Classification with CNN and ResNet-18 (Custom Architecture)

This repository presents two deep learning models for classifying images from the CIFAR-10 dataset:

1. **Baseline CNN Model** – A custom convolutional neural network built from scratch.
2. **ResNet-18 Transfer Learning** – Fine-tuned with optional architectural enhancements including attention-inspired and modular blocks.

---

## 📚 Project Overview

The CIFAR-10 dataset contains 60,000 color images (32x32 pixels) across 10 classes (airplane, car, bird, etc.). The goal is to classify these images accurately using modern deep learning techniques in PyTorch.

### This repository contains:
- **Data loaders and preprocessing**: Includes normalization and data augmentation.
- **Model architectures**: Custom CNN and ResNet-18 with additional block modules.
- **Training and evaluation scripts**.
- **Visualizations of training/testing performance**.

---

## 🧠 Model Architectures

### 1. 🔹 `1_DL_CIFAR10.ipynb` — Baseline CNN
- Custom convolutional layers, followed by ReLU activations and max-pooling.
- Trained from scratch.
- Uses CrossEntropyLoss and the Adam optimizer.
- Demonstrated accuracy: up to **~82%** on the test set.

### 2. 🔷 `deeplearning_cifar10_resnet_18_1-2.ipynb` — ResNet-18 with Optional Enhancements
This model incorporates two modular components to improve flexibility and representation power:

#### 🔸 `IntermediateBlock`:
- Contains **multiple convolutional layers** applied in parallel.
- Introduces a **learned weighting vector `a`**, calculated via a fully connected layer based on mean spatial features.
- Outputs are combined as a weighted sum, allowing the network to adaptively emphasize specific feature maps.

#### 🔸 `OutputBlock`:
- Applies **adaptive average pooling** to reduce spatial dimensions to 1×1.
- Passes through multiple fully connected layers.
- Final layer outputs logits for **10 classes**.

#### 🔸 `Model_1`:
- Stacks `IntermediateBlock` components sequentially with max-pooling between them.
- Concludes with `OutputBlock`.
- Designed for flexibility in number of convolutional layers and blocks.
- Achieved up to **85% accuracy** on CIFAR-10 with good generalization.

---

## ⚙️ Hyperparameters and Techniques

| Parameter            | Value              |
|----------------------|--------------------|
| Learning Rate        | 0.001              |
| Batch Size           | 128                |
| Epochs               | 20                 |
| Weight Decay         | 1e-4               |
| Optimizer            | Adam               |
| Activation Function  | ReLU               |
| Pooling              | MaxPooling, AdaptiveAvgPooling |
| Regularization       | Dropout, Weight Decay |
| Augmentation         | Crop, Flip, Color Jitter, Rotation |
| GPU Acceleration     | ✅ (if available)  |

---

## 🔬 Results

| Model                  | Test Accuracy (%) | Notes                      |
|------------------------|-------------------|----------------------------|
| Baseline CNN           | 82.16             | Simple CNN architecture   |
| ResNet-18 (Custom)     | 85.00             | With attention-like blocks |

Accuracy and loss plots show smooth learning with minimal overfitting.

---

## 🗃️ Dataset

- **Dataset**: [CIFAR-10]
