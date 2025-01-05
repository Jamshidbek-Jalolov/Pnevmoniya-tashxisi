# Pneumonia Image Classification

## Overview

This project aims to classify images into two categories, likely related to the presence or absence of pneumonia, using a convolutional neural network (CNN) based on the EfficientNetB0 architecture. The code is designed to work with data from a Kaggle competition named "pnevmoniya."

## Table of Contents

1. **Dependencies**
2. **Data Preparation**
3. **Model Architecture**
4. **Training**
5. **Evaluation**
6. **Submission Preparation**
7. **Instructions to Run the Code**
8. **Notes and Considerations**

## 1. Dependencies

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- pandas
- NumPy
- zipfile
- os

Ensure that all dependencies are installed in your environment before running the code.

## 2. Data Preparation

The data is assumed to be downloaded from the Kaggle competition and extracted into specific directories. The code includes functionality to download and extract the data using the Kaggle API if the API key is provided.

### Data Directories

- `train_dir`: Path to the training data directory.
- `test_dir`: Path to the test data directory.
- `sample_solution_path`: Path to the sample submission CSV file.

**Note:** Update the paths in the code to match your local file structure.

## 3. Model Architecture

The model is built using the EfficientNetB0 pre-trained on ImageNet as the base, with additional layers on top for fine-tuning:

- **Base Model:** EfficientNetB0 (frozen layers)
- **Top Layers:**
  - Global Average Pooling
  - Dense layer with 128 units and ReLU activation
  - Dropout (rate=0.5)
  - Output layer with 1 unit and sigmoid activation for binary classification

## 4. Training

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metrics:** Accuracy
- **Callbacks:** Reduce Learning Rate on Plateau
- **Epochs:** 10 ( adjustable via `EPOCHS` constant)
- **Batch Size:** 32 ( adjustable via `BATCH_SIZE` constant)

Data augmentation is applied during training to improve generalization.

## 5. Evaluation

The model is evaluated on the validation set using the following metrics:

- **Classification Report:** Precision, Recall, F1-Score, and Support for each class.
- **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.
- **ROC Curve and AUC:** To assess the trade-off between true positive rate and false positive rate.

## 6. Submission Preparation

Predictions are made on the test set, and a submission CSV file is generated with image filenames and predicted labels.

## 7. Instructions to Run the Code

1. **Set Up Kaggle API (Optional but Recommended):**
   - Place your `kaggle.json` file in the current directory to automatically download the data.
   - Ensure the Kaggle API is installed and properly configured.

2. **Update Data Paths:**
   - Modify `train_dir`, `test_dir`, and `sample_solution_path` in the code to point to your data directories.

3. **Run the Code:**
   - Execute the script in a Python environment with all dependencies installed.
   - The code will perform data loading, model training, evaluation, and submission preparation sequentially.

4. **Results:**
   - The trained model's performance is printed in the console.
   - A ROC curve plot is saved as `roc_curve.png`.
   - The submission file is saved as `submission.csv`.

## 8. Notes and Considerations

- **Data Augmentation:** Applied only to the training data to prevent information leakage.
- **Model Fine-Tuning:** The base model layers are frozen; consider fine-tuning for better performance.
- **Computational Resources:** Training deep learning models requires significant computational power; consider using GPUs.
- **License:** This code is provided under the MIT License. Feel free to modify and use it for your purposes.
