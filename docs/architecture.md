# System Architecture

## Overview

This project implements an end-to-end machine learning pipeline for early autism detection using machine learning techniques.

The system takes raw screening data, performs preprocessing, trains a classification model, evaluates performance, and generates predictions for new inputs.


## Pipeline Architecture
Raw Dataset
|
↓
Data Preprocessing
|
├── Data Cleaning
├── Feature Transformation
├── Label Encoding
└── Outlier Handling
|
↓
Feature Dataset (X, y)
|
↓
Train-Test Split
|
↓
SMOTE (Class Balancing)
|
↓
Random Forest Classifier
|
↓
Model Evaluation
|
├── Accuracy
├── Confusion Matrix
└── Classification Report
|
↓
Saved Model (.pkl)
|
↓
Prediction System 




## Project Components

### 1. Data Layer

Location:
data/raw/


Contains the original dataset used for training.

The dataset includes:

- Screening attributes (A1-A10 scores)
- Demographic information
- Medical history related features
- Target variable (Class/ASD)

### 2. Preprocessing Layer

File:


src/data_preprocessing.py


Responsibilities:

- Loading dataset
- Removing unnecessary columns
- Handling inconsistent categorical values
- Encoding categorical features
- Handling outliers
- Preparing features and target variables


### 3. Training Layer

File:


src/train.py


Responsibilities:

- Splitting training and testing data
- Handling class imbalance using SMOTE
- Training Random Forest classifier
- Saving trained model


Output:


models/autism_model.pkl 

### 4. Evaluation Layer

File:


src/evaluate.py


Responsibilities:

- Loading trained model
- Generating predictions
- Evaluating model performance using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix


### 5. Prediction Layer

File:


src/predict.py


Responsibilities:

- Loading trained model
- Loading saved encoders
- Transforming new input data
- Generating ASD prediction


## Model Storage

Trained artifacts are stored in:
models/
│
├── autism_model.pkl
└── encoders.pkl



## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Seaborn


