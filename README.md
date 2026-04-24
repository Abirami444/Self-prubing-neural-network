# Self-Pruning Neural Network on CIFAR-10

## 1. Overview

This project presents the implementation of a self-pruning neural network that learns to eliminate redundant weights during training. Unlike conventional pruning approaches that operate post-training, this method integrates pruning into the learning process through the use of learnable gate parameters associated with each weight.

The system combines a convolutional feature extractor with prunable fully connected layers, enabling the model to retain only the most relevant connections while maintaining competitive classification performance.

---

## 2. Core Concept

Each weight in the prunable layers is associated with a learnable gate value:

* Gate value close to 1: the weight remains active
* Gate value close to 0: the weight is effectively pruned

The training objective is defined as:

Total Loss = Classification Loss + λ × Sparsity Loss

Where:

* Classification Loss: Cross-Entropy Loss for supervised learning
* Sparsity Loss: L1 norm applied to gate values
* λ (lambda): hyperparameter controlling the sparsity–accuracy trade-off

---

## 3. Theoretical Justification

L1 regularization encourages sparsity by applying a constant gradient that drives parameters toward zero. This property enables decisive elimination of less important weights. In contrast, L2 regularization produces gradual shrinkage without achieving exact zero values, making it less suitable for pruning objectives.

---

## 4. Model Architecture

The network consists of two primary components:

### 4.1 Convolutional Feature Extractor

* Conv2d (3 → 32), BatchNorm, ReLU, MaxPool
* Conv2d (32 → 64), BatchNorm, ReLU, MaxPool
* Conv2d (64 → 128), BatchNorm, ReLU, MaxPool

This stage extracts hierarchical spatial features from CIFAR-10 images.

### 4.2 Prunable Fully Connected Layers

* PrunableLinear (2048 → 512) + BatchNorm + ReLU + Dropout
* PrunableLinear (512 → 10)

These layers determine which feature combinations are necessary for classification.

---

## 5. Key Implementation Details

### 5.1 PrunableLinear Layer

Each weight is paired with a learnable parameter (`gate_score`). The gate is computed using the sigmoid function.

* During training: soft gating is applied to maintain differentiability
* During inference: hard thresholding converts gates into binary values (0 or 1), enabling true pruning

### 5.2 Gate Initialization

Gate parameters are initialized to a positive value (e.g., +3.0), corresponding to an initial sigmoid output close to 1. This ensures that all connections start active, and sparsity is introduced gradually through regularization.

### 5.3 Sparsity Loss Normalization

The sparsity loss is computed as the mean of all gate values rather than the sum. This normalization ensures that the magnitude of the sparsity term remains consistent regardless of model size.

### 5.4 Separate Learning Rates

Different learning rates are assigned to weights and gate parameters:

* Model weights: standard learning rate (e.g., 1e-3)
* Gate parameters: higher learning rate (e.g., 5e-3)

This allows gate values to converge more rapidly toward binary decisions.

---

## 6. Experimental Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 0.1    | 77.25             | 60.85              |
| 0.3    | 75.17             | 73.36              |
| 0.6    | 72.06             | 83.46              |

---

## 7. Analysis

The results demonstrate a clear trade-off between model accuracy and sparsity:

* Increasing λ leads to higher sparsity levels
* Excessive sparsity results in degradation of classification accuracy
* A moderate value (λ = 0.3) achieves a balanced trade-off, maintaining strong accuracy while significantly reducing the number of active parameters

---

## 8. Output Visualizations

### 8.1 Gate Distribution

The distribution of gate values shows a concentration near zero (pruned weights) and near one (retained weights), indicating effective separation between important and redundant connections.

### 8.2 Accuracy–Sparsity Trade-off

A dual-axis plot illustrates how increasing sparsity affects model performance.

All visual outputs are stored in the `outputs/` directory.

---

## 9. Model Compression

After training, pruned weights are physically removed to construct a compressed model. This results in a reduced parameter count and improved inference efficiency without requiring gate computations during deployment.

---

## 10. Execution Instructions

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

Run the training script:

```bash
python self_pruning_nn_v3.py
```

The CIFAR-10 dataset is automatically downloaded during the first run.

---

## 11. Project Structure

self-pruning-neural-network/
├── self_pruning_nn_v3.py
├── README.md
└── outputs/
├── gate_distribution.png
└── accuracy_sparsity_tradeoff.png

---

## 12. Environment

* Python 3.11
* PyTorch
* torchvision
* NumPy, Matplotlib
* GPU: NVIDIA Tesla T4 (Kaggle environment)

---

## 13. Future Work

* Extension to structured pruning (e.g., channel-level pruning)
* Evaluation of compressed model inference latency
* Exploration of L0 regularization techniques
* Implementation of progressive sparsity scheduling
* Integration with knowledge distillation methods

---

## 14. Author

Abirami
