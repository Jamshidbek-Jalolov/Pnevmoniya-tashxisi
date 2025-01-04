
# Pneumonia Detection using ResNet50

This repository contains the implementation of a deep learning model to classify X-ray images into two categories: Pneumonia and Normal. The model is built using the ResNet50 architecture, which is fine-tuned to achieve high accuracy on the Pneumonia dataset.

## Project Overview

- **Dataset**: The dataset used for training and testing the model is provided by the Kaggle competition 'Pnevmoniya'. It consists of chest X-ray images categorized into two classes: "Normal" and "Pneumonia".
- **Model Architecture**: The model is based on the pre-trained ResNet50, which is fine-tuned to improve performance for this specific task.
- **Data Augmentation**: Data augmentation techniques such as rotation, width/height shift, zoom, and horizontal flip are applied to increase the robustness of the model.
- **Fine-tuning**: The model undergoes fine-tuning by unfreezing the layers of ResNet50 after initial training.
- **Metrics**: The model is evaluated using accuracy, confusion matrix, classification report, and ROC-AUC score.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- pandas
- scikit-learn
- matplotlib
- Kaggle API

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Kaggle API credentials:
   - Download your `kaggle.json` from your Kaggle account (Account > API > Create New API Token).
   - Place `kaggle.json` in the `~/.kaggle/` directory (or use the provided setup in the code).

## Usage

1. Download the dataset using the Kaggle API by running the following command:
   ```bash
   kaggle competitions download -c pnevmoniya
   ```

2. Extract the dataset:
   ```bash
   unzip pnevmoniya.zip -d /content/pnevmoniya/
   ```

3. Run the training script:
   ```python
   python train_model.py
   ```

4. The training process involves two phases:
   - Initial training with the frozen layers of the ResNet50 model.
   - Fine-tuning the ResNet50 model by unfreezing all layers.

5. After training, the model is evaluated using test data, and the predictions are saved to a CSV file for submission.

## Output

- **Training History Plots**: Plots showing the training and validation accuracy and loss over epochs.
- **ROC Curve**: A graph representing the True Positive Rate against the False Positive Rate.
- **Classification Report**: Precision, Recall, F1-Score, and Support for both classes.
- **Confusion Matrix**: Visual representation of true vs predicted labels.

## Submission

The model's predictions on the test dataset are saved in a CSV file (`submission.csv`) for submission to the Kaggle competition.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

Feel free to customize it further for your needs! Let me know if you'd like any adjustments.
