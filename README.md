# Early Autism Screening ML System 🧠

An end-to-end machine learning pipeline for early autism screening using machine learning techniques. The project focuses on data preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, and prediction using a trained classification model.

---

## 📌 Project Overview

Early detection of Autism Spectrum Disorder (ASD) can help in timely intervention and support. This project develops a machine learning classification system that predicts ASD likelihood based on behavioral screening responses and demographic features.

The project implements a complete ML workflow from raw data processing to model prediction.

---

## 🚀 Features

- Complete machine learning pipeline
- Data cleaning and preprocessing
- Handling categorical features using Label Encoding
- Outlier detection and treatment
- Class imbalance handling using SMOTE
- Multiple model experimentation
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation using multiple metrics
- Saved model and encoder artifacts
- Prediction pipeline for new inputs

---

## 🏗️ Project Architecture

```
Raw Dataset
     |
     ↓
Data Preprocessing
     |
     ├── Data Cleaning
     ├── Feature Encoding
     ├── Outlier Handling
     |
     ↓
Train-Test Split
     |
     ↓
SMOTE (Class Balancing)
     |
     ↓
Model Training
     |
     ↓
Hyperparameter Optimization
     |
     ↓
Model Evaluation
     |
     ↓
Saved Model (.pkl)
     |
     ↓
Prediction System
```

For detailed architecture:
[Architecture Documentation](docs/architecture.md)

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn

## Development Tools

- Jupyter Notebook
- VS Code
- Git

---

# 📂 Project Structure

```
ml-pipeline/

│
├── data/
│   └── raw/
│       └── train.csv
│
├── models/
│   ├── autism_model.pkl
│   └── encoders.pkl
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── docs/
│   ├── architecture.md
│   └── ml_report.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📊 Machine Learning Workflow

## 1. Data Preprocessing

Performed preprocessing steps:

- Converted data types
- Removed unnecessary columns:
  - ID
  - age_desc
- Corrected inconsistent categorical values
- Handled missing/unknown categorical values
- Applied Label Encoding
- Detected and handled outliers using IQR method

---

## 2. Handling Class Imbalance

The dataset contained imbalanced target classes.

SMOTE (Synthetic Minority Oversampling Technique) was applied on training data to improve model learning.

---

## 3. Model Training

Multiple classification algorithms were evaluated:

| Model | Purpose |
|---|---|
| Decision Tree | Baseline classifier |
| Random Forest | Ensemble learning |
| XGBoost | Gradient boosting approach |

---

## 4. Hyperparameter Optimization

RandomizedSearchCV was used to find optimal parameters for each model.

The final model was selected based on cross-validation performance.

---

# 🏆 Final Model

The best performing model:

```
Random Forest Classifier
```

Best Parameters:

```text
n_estimators = 50
max_depth = 20
bootstrap = False
random_state = 42
```

---

# 📈 Model Performance

Cross-validation Accuracy:

```
92.52%
```

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Detailed report:

[ML Performance Report](docs/ml_report.md)

---

# 💾 Saved Model Artifacts

After training, the following files are generated:

```
models/

├── autism_model.pkl
└── encoders.pkl
```

`autism_model.pkl`
- Contains the trained Random Forest model.

`encoders.pkl`
- Contains Label Encoders required for transforming new input data.

---

# ⚙️ Installation & Setup

## Clone Repository

```bash
git clone <repository-url>

cd ml-pipeline
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Pipeline

## Train Model

```bash
python src/train.py
```

---

## Evaluate Model

```bash
python src/evaluate.py
```

---

## Make Prediction

```bash
python src/predict.py
```

---

# 🔮 Future Improvements

- Deploy model using FastAPI
- Add frontend interface for predictions
- Containerize using Docker
- Deploy on cloud platforms
- Use advanced ML pipelines with Scikit-learn Pipeline

---

# 📚 Documentation

Additional documentation:

- [System Architecture](docs/architecture.md)
- [Machine Learning Report](docs/ml_report.md)

---

# 👨‍💻 Author

**Shivam Raj**

---

# 📄 License

This project is licensed under the MIT License.
