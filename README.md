# Melanoma-Classification-Using-Custom-CNN

## Overview
### Melanoma is a severe type of skin cancer. This project involves building a custom Convolutional Neural Network (CNN) to classify 9 categories of skin diseases, including melanoma, from a dataset of 2357 images. This project does not use pre-trained models and implements the solution from scratch.

## Problem Statement
### The goal of this project is to build a CNN-based model that can accurately classify melanoma from images of skin lesions. Early detection of melanoma can significantly reduce mortality rates, and an automated solution can assist dermatologists in diagnosis.

## Project Steps
### 1. Data Reading and Understanding
#### Understanding and reading the dataset containing images of different skin conditions.

### 2. Dataset Creation
#### Organizing and preparing the dataset into categories.

### 3. Dataset Visualization
#### Visualizing a sample of the data to understand the types and distributions.

### 4. Initial Model Building and Training
#### Creating a CNN model and training it on the dataset.

### 5. Data Augmentation
#### Applying transformations to augment the data and improve model performance.

### 6. Class Distribution Analysis
#### Analyzing the class distribution to check for any imbalances.

### 7. Handling Class Imbalances
#### Techniques such as oversampling, undersampling, or class weights may be applied to handle class imbalances.

### 8. Final Model Training
#### Training the final model on the processed dataset.

### 9. Evaluation and Results
#### Evaluating the model on test data and reviewing its performance.

### 10. Conclusion
#### Summarizing the findings, challenges, and future steps.

## Requirements
### Make sure to install the following Python packages for this project:
### * tensorflow
### * numpy
### * matplotlib
### * sklearn

### You can install them using pip:
### * pip install tensorflow numpy matplotlib scikit-learn

## Data
### The dataset used in this project is the ISIC (International Skin Imaging Collaboration) dataset, which includes images of different types of skin lesions.

### Train Directory: Contains images for training the model, organized into 9 categories.
### Test Directory: Contains images for testing the model's performance.

## Instructions
### Clone or download this repository.
### Make sure the dataset is in the correct directory as specified in the notebook (TRAIN_DIR and TEST_DIR).
### Run the Jupyter notebook step by step to train the model and evaluate the results.

## Evaluation
### After training, the model is evaluated using accuracy, precision, recall, and F1 score metrics. It also provides a confusion matrix to help analyze the performance across different classes.

## Conclusion
### This project demonstrates the process of building a custom CNN for melanoma classification. With further fine-tuning, augmentation, and hyperparameter optimization, it can potentially be enhanced to perform well in real-world diagnostic scenarios.
