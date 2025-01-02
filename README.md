# Pnevmoniya-tashxisi

Quyida ushbu loyiha uchun README fayl matni keltirilgan:

---

# Pneumonia Detection using ResNet34

## Overview
This project utilizes the ResNet34 (via ResNet50) deep learning architecture to classify chest X-ray images as either **Normal** or **Pneumonia**. The dataset consists of X-ray images of pediatric patients aged 1 to 5 years, divided into training and test datasets.

The project outputs predictions in a CSV file format, compatible with the provided sample solution.

---

## Project Steps

### 1. Data Preparation
- **Training Data**: X-ray images are categorized into two folders: `train/normal` and `train/pneumonia`.
- **Testing Data**: X-ray images are stored in a folder, with no subcategories.
- Image preprocessing includes resizing to \(224 \times 224\), normalization to [0, 1], and augmentation (e.g., rotation, flipping).

### 2. Model Architecture
- ResNet34 is implemented using the ResNet50 architecture provided by TensorFlow/Keras.
- The base ResNet50 layers are frozen initially to leverage pre-trained weights.
- The top layers are modified to include:
  - A `GlobalAveragePooling2D` layer.
  - A dense layer with 256 units and ReLU activation.
  - A final dense layer with a sigmoid activation for binary classification.

### 3. Training and Fine-Tuning
- The model is trained in two stages:
  - Initial training with frozen base layers.
  - Fine-tuning with all layers trainable.
- Loss function: `Binary Crossentropy`.
- Optimizer: `Adam`.

### 4. Testing and Prediction
- The test dataset contains X-ray images without labels.
- Predictions are made using the trained model and saved in the required CSV format.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.0+
- NumPy, Pandas, Matplotlib, Scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pneumonia-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model
1. Place training data in the `train/` folder:
   - `train/normal`
   - `train/pneumonia`
2. Run the training script:
   ```bash
   python train.py
   ```

### Testing the Model
1. Place test images in the `test/` folder.
2. Run the prediction script:
   ```bash
   python predict.py --test_dir /path/to/test/
   ```

### Results
Predictions are saved in a CSV file (`predicted_solution.csv`) in the following format:
| id                   | labels |
|----------------------|--------|
| test_img_10001.jpeg  | 1      |
| test_img_10002.jpeg  | 0      |

---

## Output
- Training and validation accuracy/loss plots.
- A confusion matrix and classification report for evaluation.
- A CSV file containing predictions for the test dataset.

---

