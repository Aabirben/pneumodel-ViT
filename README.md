# Pneumonia Detection using Vision Transformer (ViT) - Dataset Preprocessing and Model Development

## Project Overview
This project develops a Vision Transformer (ViT) model to classify chest X-ray images as Normal or Pneumonia, targeting a test accuracy of ≥85%. It includes dataset preprocessing, model fine-tuning, and evaluation to ensure robust performance for medical applications.

## Dataset Preprocessing
### 1. Initial Dataset Overview
- Dataset located at `/content/drive/MyDrive/pneumonia-ViT3/dataset/`.
- Split into train, validation, and test sets.
- Classes: Normal and Pneumonia.
- Initial imbalance:
  - Train: 1341 Normal, 3875 Pneumonia
  - Validation: 8 Normal, 8 Pneumonia
  - Test: 234 Normal, 390 Pneumonia
- Imbalance risks biasing the model toward Pneumonia.

### 2. Data Augmentation for Balancing
- Oversampled Normal images in train and validation sets to balance classes.
- Post-balancing:
  - Train+Validation: 3883 Normal, 3883 Pneumonia (7766 images total)
  - Test unchanged: 234 Normal, 390 Pneumonia (624 images)
- Ensures robust training while testing reflects real-world distribution.

### 3. Preprocessing Pipeline
- Each image processed as follows:
  - Resized to 224x224 pixels.
  - Converted to RGB.
  - Normalized using dataset-specific mean and standard deviation.
- Augmentation for train set: random horizontal flips, rotations (±15°), translations, scaling, color adjustments, random cropping.
- Validation and test sets: only resizing and normalization.
- Saved processed images as metadata files (`train_val.csv`, `test.csv`) under `/content/drive/MyDrive/pneumonia-ViT3/dataset/processed_dataset_v7`.

### 4. Validation and Output
- Visual checks confirmed proper preprocessing and augmentation.
- Final dataset: 7766 balanced train+validation images, 624 imbalanced test images, all 224x224 RGB.
- Ensures compatibility with ViT and enhances feature extraction.

## Model Development
### 1. Model Architecture
- Base model: ViT pre-trained on ImageNet-21k.
- Input size: 224x224 RGB images.
- Uses 12 transformer layers with 16x16 patches, ~86M parameters.

### 2. Custom Classifier Layers
- Replaced final classifier layer with a new layer for binary classification (Normal vs. Pneumonia).
- Added dropout (0.5) in classifier and (0.2) in hidden and attention layers to reduce overfitting.
- All layers fine-tuned to adapt to medical images.
- Applied balanced class weights to handle any residual imbalance.

### 3. Training Details
- Optimizer: Adam.
- Loss: Cross-entropy.
- Metric: Accuracy.
- Trained for 15 epochs with batch size 8 (train) and 4 (validation), using gradient accumulation.
- Learning rate: 1e-6 with reduce-on-plateau scheduler.
- Regularization: Weight decay (0.15) and dropout.
- Mixed precision enabled for efficiency.
- Model saved at `/content/drive/MyDrive/pneumonia-ViT3/models/pneumonia_chest_xray_vit_v8`.

### 4. Addressing Class Imbalance Bias
- Oversampling balanced training data, improving Normal class detection.
- Dropout and weight decay reduced overfitting.
- Model performs robustly on imbalanced test set, with balanced precision and recall.

## Results
- Validation:
  - Accuracy: 94.59%
  - F1-score: 94.58%
- Test:
  - Accuracy: 89.26% (meets ≥85% goal)
  - F1-score: 88.28%
- Classification Report (Test):

  | Class     | Precision | Recall  | F1-score | Support |
  |-----------|-----------|---------|----------|---------|
  | Normal    | 0.8995    | 0.8034  | 0.8488   | 234     |
  | Pneumonia | 0.8892    | 0.9462  | 0.9168   | 390     |
  | Accuracy  |           |         | 0.8926   | 624     |
  | Macro Avg | 0.8943    | 0.8748  | 0.8828   | 624     |

- Visualization: Displayed 6 test images; most correctly classified.

## How to Use
### Dataset Preparation
- Place dataset in `/content/drive/MyDrive/pneumonia-ViT3/dataset/`.
- Run preprocessing script to generate metadata files in `processed_dataset_v7`.

### Model Training
- Use provided training script.
- Monitor validation metrics and save checkpoints.

### Evaluation
- Evaluate on test set without augmentation for real-world performance.
- Analyze precision, recall, and accuracy metrics.

## Contact
For questions or support, contact the project team  Pneumodel at [iaabirbenhamamouche@gmail.com].

