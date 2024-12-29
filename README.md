# README

## Handwritten Signature Verification using Siamese Neural Network

This project implements a Siamese Neural Network (SNN) for verifying handwritten signatures using datasets like **BHSig260** (Bengali and Hindi) and **CEDAR**. The model is designed to distinguish between genuine and forged signatures through contrastive loss and transfer learning techniques with the VGG16 architecture.

---

## Project Overview

The goal is to develop an automated system that can identify and verify signatures based on similarity metrics. It uses Siamese Neural Networks to compare signature pairs and determine whether they match (genuine) or not (forged).

---

## Key Features

- **Preprocessing:** Converts images to grayscale, applies Gaussian blur, and performs thresholding to highlight signatures.
- **Transfer Learning:** Leverages VGG16 pre-trained model for feature extraction.
- **Data Handling:** Automatically loads datasets, generates positive and negative pairs, and splits data into training, validation, and testing subsets.
- **Loss Function:** Uses contrastive loss to optimize the SNN for comparing signature pairs.
- **Performance Metrics:** Evaluates the model with metrics such as accuracy and loss.
- **Visualizations:** Plots training and validation accuracy/loss over epochs.

---

## Datasets

This project uses three signature datasets:

1. **BHSig260-Bengali**
2. **BHSig260-Hindi**
3. **CEDAR**

---

## Setup Instructions

1. **Environment Requirements:**

   - Python 3.7+
   - TensorFlow 2.x
   - OpenCV
   - NumPy
   - Matplotlib
   - Kaggle API (for dataset download)

2. **Dataset Download:**
   Install `kagglehub` and download datasets using:

   ```python
   path = kagglehub.dataset_download("ishanikathuria/handwritten-signature-datasets")
   ```

3. **Dependencies:**
   Install required Python libraries:

   ```bash
   pip install tensorflow opencv-python-headless numpy matplotlib
   ```

4. **Run the Code:**
   Execute the Python script to preprocess the data, train the model, and evaluate performance on the test sets.

---

## Code Structure

### 1. **Preprocessing**

- Resizes images to 224x224 (input size for VGG16).
- Applies grayscale conversion, Gaussian blur, and thresholding.
- Prepares pairs for model training.

### 2. **Model Architecture**

- A Siamese network with two VGG16 branches sharing weights.
- A custom distance metric using the Euclidean distance.
- Trains with contrastive loss to minimize the distance between similar pairs and maximize it for dissimilar pairs.

### 3. **Data Generators**

- Creates TensorFlow datasets for training, validation, and testing.
- Ensures efficient batching and prefetching.

### 4. **Training and Evaluation**

- Includes callbacks for early stopping and learning rate reduction.
- Evaluates model performance across Bengali, Hindi, and CEDAR datasets.

---

## Results

- The model achieves distinct accuracies for each dataset, demonstrating its ability to generalize across languages and data types.
- Visualization of accuracy and loss curves provides insight into model convergence and validation performance.

---

## Notes

- Ensure the dataset structure matches the expected hierarchy in the code.
- Modify the `load_dataset` function to handle additional datasets or variations in directory structure.

---

## Future Enhancements

- Incorporate additional datasets to improve model generalization.
- Experiment with data augmentation techniques for better robustness.
- Optimize model architecture to improve computational efficiency.

---

## Acknowledgments

- Datasets courtesy of the **BHSig260** and **CEDAR** projects.
- Pretrained VGG16 model from **Keras Applications**.

---

## Badges

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

