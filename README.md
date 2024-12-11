# CNN-Based Image Classification

This repository contains multiple implementations of Convolutional Neural Networks (CNNs) for image classification tasks. The models range from custom-built CNN architectures to transfer learning using pre-trained networks like ResNet50. This repository is structured for flexibility, enabling experimentation with different architectures and datasets.


## Features

1. **Custom CNN Models:**
   - Designed from scratch for specific datasets.
   - Includes feature extraction using multiple convolutional and pooling layers.
   - Augmentation and regularization to prevent overfitting.

2. **Small CNN Model:**
   - A simplified CNN architecture suitable for small datasets or quick experiments.
   - Focuses on achieving a balance between performance and computational efficiency.

3. **Transfer Learning:**
   - Uses ResNet50 as the base model for robust feature extraction.
   - Custom dense layers added for classification tasks.
   - Includes fine-tuning for improved performance.

4. **Data Augmentation:**
   - Rotation, zoom, and flip augmentations implemented to enhance dataset variability.
   - Real-time augmentation during training using `ImageDataGenerator`.

5. **Evaluation:**
   - Supports metrics like accuracy, confusion matrices, and loss/accuracy plots.
   - Clear separation of train, validation, and test datasets to prevent data leakage.

6. **Callbacks:**
   - Early stopping to prevent overfitting.
   - Learning rate reduction on plateau.

---

## Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

Install the required dependencies with:
```bash
pip install -r requirements.txt
```

---

## Directory Structure
```
.
├── data
│   ├── train
│   ├── validation
│   └── test
├── models
│   ├── custom_cnn.py
│   ├── small_cnn.py
│   ├── transfer_learning_resnet50.py
├── notebooks
│   └── training_visualization.ipynb
├── requirements.txt
├── README.md
└── utils
    ├── data_preprocessing.py
    └── plot_metrics.py
```

---

## Getting Started

### 1. Prepare the Dataset

Organize the dataset into the following structure:
```
.
├── train
│   ├── class_1
│   ├── class_2
│   └── ...
├── validation
│   ├── class_1
│   ├── class_2
│   └── ...
└── test
    ├── class_1
    ├── class_2
    └── ...
```

### 2. Train a Model

Run any of the scripts in the `models/` directory. For example, to train the custom CNN model:
```bash
python models/custom_cnn.py
```

### 3. Evaluate the Model

Visualize metrics and evaluate model performance using the provided utilities. For example:
```bash
python utils/plot_metrics.py
```

---

## Models Overview

### Custom CNN Model
A robust architecture with multiple convolutional layers, batch normalization, and dropout for regularization.

### Small CNN Model
A lightweight architecture ideal for quick experimentation and smaller datasets.

### Transfer Learning with ResNet50
A state-of-the-art approach leveraging pre-trained weights of ResNet50 for feature extraction and fine-tuning.

## Results

Include details about the performance of different models, such as:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrices

## Contributing

Feel free to fork this repository and submit pull requests. Contributions in the form of bug fixes, feature additions, or performance enhancements are welcome.



