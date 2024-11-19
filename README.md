# finalaml
final Resnet18 on diabet rinopathy detection
---

# Diabetic Retinopathy Detection

This project focuses on detecting different stages of diabetic retinopathy using deep learning models. A pretrained baseline model (ResNet18) and an enhanced model with Batch Normalization and Dropout layers were implemented, trained, and evaluated for performance improvement.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models](#models)
   - [Baseline Model](#baseline-model)
   - [Enhanced Model](#enhanced-model)
4. [Results](#results)
5. [Setup Instructions](#setup-instructions)
6. [How to Run](#how-to-run)
7. [Project Files](#project-files)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project was to classify retinal images into one of five classes:
- No_DR: No Diabetic Retinopathy.
- Mild: Mild Diabetic Retinopathy.
- Moderate: Moderate Diabetic Retinopathy.
- Severe: Severe Diabetic Retinopathy.
- Proliferate_DR: Proliferate Diabetic Retinopathy.

Key Objectives:
- Train a baseline ResNet18 model.
- Enhance the model by adding Batch Normalization and Dropout layers to reduce overfitting.
- Compare the performance of both models using metrics such as accuracy, precision, recall, and F1-score.

---

## Dataset

- Source: [Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
- Structure: Images categorized into folders based on the severity of diabetic retinopathy.
- Train-Test Split: 
  - 80% Training
  - 20% Testing

---

## Models

### Baseline Model
- Architecture: Pretrained ResNet18.
- Modifications:
  - Replaced the final fully connected layer to output predictions for five classes.
- Loss Function: CrossEntropy Loss.
- Optimizer: Adam Optimizer with a learning rate of 0.001.

### Enhanced Model
- Architecture: Pretrained ResNet18 with:
  - Batch Normalization to stabilize learning.
  - Dropout (50%) to reduce overfitting.
  - Additional fully connected layer for richer feature extraction.
- Loss Function: CrossEntropy Loss.
- Optimizer: Adam Optimizer with a learning rate of 0.001.

---

## Results

### Baseline Model
- Training Accuracy: 96.6% after 10 epochs.
- Testing Accuracy: 78% after 10 epochs.
- Observations: High training accuracy but signs of overfitting with a large gap in test metrics.

### Enhanced Model
- Training Accuracy: 91.77% after 10 epochs.
- Testing Accuracy: 74% after 10 epochs.
- Observations: Enhanced regularization techniques (Batch Normalization and Dropout) mitigated overfitting and improved generalization.

### Metrics Comparison
- Metrics such as accuracy, precision, recall, and F1-score were evaluated and visualized across epochs.

![Model Accuracy Comparison](insert-accuracy-comparison-image-path-here)
![Model Precision Comparison](insert-precision-comparison-image-path-here)

---

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher.
- Install the required libraries:
  
  pip install -r requirements.txt
  
  *(Create a requirements.txt file with your dependencies such as torch, torchvision, matplotlib, scikit-learn, etc.)*

### 2. Clone the Repository
git clone https://github.com/<your_username>/<repository_name>.git
cd <repository_name>

### 3. Download Dataset
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered) and place it in the data/ directory.

### 4. Setup the Dataset
Organize the dataset folder as follows:
data/
    gaussian_filtered_images/
        No_DR/
        Mild/
        Moderate/
        Severe/
        Proliferate_DR/

---

## How to Run

1. Train the Baseline Model
   
   python baseline_model.py
   

2. Train the Enhanced Model
   
   python enhanced_model.py
   

3. Evaluate Models
   - Metrics such as accuracy, precision, recall, and F1-score will be displayed and saved as plots.

4. Visualize Results
   - Check output/ folder for graphs comparing both models' performance.

---
## Project Files

- baseline_model.py: Code for training the baseline ResNet18 model.
- enhanced_model.py: Code for training the enhanced model with Batch Normalization and Dropout.
- data/: Contains the dataset (not included due to size).
- output/: Contains the generated visualizations and model evaluation results.
- README.md: Project description and instructions.
- requirements.txt: List of dependencies.

---

## Acknowledgments

- Dataset: [Diabetic Retinopathy Dataset by Sovit Rath on Kaggle](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
- Frameworks: PyTorch, Torchvision

---

Replace placeholders (e.g., insert-accuracy-comparison-image-path-here) with actual paths to images in your repository. Let me know if you need additional edits!
