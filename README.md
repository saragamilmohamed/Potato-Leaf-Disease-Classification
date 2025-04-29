

# Potato Leaf Disease Classification

This project focuses on classifying potato leaf images into three categories: **Early Blight**, **Late Blight**, and **Healthy**. Two approaches are implemented and compared:

- **Custom CNN Model**: Achieves **98% accuracy**.
- **Pretrained VGG19 Model**: Achieves **97% accuracy**.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Models](#models)
   - [Custom CNN](#custom-cnn)
   - [Pretrained VGG19](#pretrained-vgg19)
4. [Results](#results)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Directory Structure](#directory-structure)
   - [Training](#training)
   - [Evaluation & Inference](#evaluation--inference)
6. [Usage Examples](#usage-examples)
7. [Notes](#notes)
8. [License](#license)

## Overview

Potato crops are susceptible to diseases that can significantly impact yield. Early detection through image-based classification can aid in timely intervention. This project implements:

- A **Custom CNN** trained from scratch.
- A **Transfer Learning** approach using a pretrained VGG19 model.

Both models are evaluated on a test set of high-resolution potato leaf images.

## Dataset

- **Source**: Kaggle "Potato Disease Leaf Dataset" (3 classes, 256×256 images).
- **Split**:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%
- **Classes**:
  1. Early Blight
  2. Late Blight
  3. Healthy

## Models

### Custom CNN

- **Architecture**:
  - Input → Rescaling(1/255) → Data Augmentation → 5× Conv2D+MaxPool layers → Flatten → Dense(64) → Output(3, softmax)
- **Training**:
  - Optimizer: Adam
  - Loss: Sparse Categorical Crossentropy
  - Batch size: 32
  - Epochs: 50
- **Accuracy**: **98%** on test set

### Pretrained VGG19

- **Architecture**:
  - Input → Data Augmentation → `tf.keras.applications.vgg19.preprocess_input` → VGG19(base, frozen) → GlobalAveragePooling → Dropout(0.2) → Dense(128) → Dense(32) → Output(3, softmax)
- **Training**:
  - Optimizer: Adam
  - Loss: Sparse Categorical Crossentropy
  - Batch size: 32
  - Epochs: 60 (with early stopping)
- **Accuracy**: **97%** on test set

## Results

| Model            | Test Accuracy |
|------------------|---------------|
| Custom CNN       | **98%**       |
| Pretrained VGG19 | **97%**       |

The Custom CNN slightly outperforms the transfer-learning model, likely due to the specific features of the dataset.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- matplotlib, pandas
- streamlit (for optional web demo)

Install dependencies:

```bash
pip install tensorflow scikit-learn matplotlib pandas 
```

### Directory Structure

```
project/
├── data/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
├── models/
│   ├── custom_cnn.h5
│   └── vgg19_transfer.h5
├── scripts/
│   ├── train_custom_cnn.py
│   ├── train_vgg19.py
│   └── evaluate.py
├── app.py            # streamlit demo
├── processing.py     # preprocessing utilities
└── README.md
```

## Usage Examples

Inside a Python REPL:

```python
from tensorflow.keras.models import load_model
import numpy as np, os
from tensorflow.keras.preprocessing import image

model = load_model('models/custom_cnn.h5')
img = image.load_img('custom.jpg', target_size=(256,256))
arr = image.img_to_array(img) / 255.0
pred = model.predict(np.expand_dims(arr,0))
print('Predicted:', ['Early Blight','Healthy','Late Blight'][np.argmax(pred)])
```

## Notes

- Data augmentation is only applied during training.
- Validation/Test images are only rescaled.
- Adjust hyperparameters (`learning_rate`, `batch_size`, `epochs`) in the training scripts for further tuning.



