# Machine Learning Model Report

## Project Title

Early Autism Detection using Machine Learning


## Objective

The objective of this project is to develop a machine learning classification system that predicts the possibility of Autism Spectrum Disorder (ASD) based on screening and demographic features.


## Dataset

The dataset contains screening questionnaire responses and demographic information.

Features include:

- A1-A10 behavioral screening scores
- Age
- Gender
- Ethnicity
- Medical history features
- Country information

Target variable:

where:

- 0 → No ASD
- 1 → ASD


## Data Preprocessing

The following preprocessing steps were applied:

### Data Cleaning

- Removed unnecessary columns:
    - ID
    - age_desc

- Corrected inconsistent categorical values.

Examples:

- Viet Nam → Vietnam
- Hong Kong → China


### Missing Value Handling

Unknown categorical values were grouped into meaningful categories.

Examples:

- "?"
- "others"

were converted into:



### Feature Encoding

Categorical features were transformed into numerical format using:



### Outlier Handling

Outliers in numerical features were detected using the IQR method and replaced using median values.


## Handling Class Imbalance

The dataset contained imbalanced target classes.

To improve model learning:


was applied on training data.


## Models Evaluated

Three classification models were experimented with:

1. Decision Tree Classifier

2. Random Forest Classifier

3. XGBoost Classifier


## Hyperparameter Optimization

RandomizedSearchCV was used for hyperparameter tuning.

The best performing model was selected based on cross-validation accuracy.


## Final Model

Algorithm:
Random Forest Classifier

Best Parameters:
n_estimators = 50

max_depth = 20

bootstrap = False

random_state = 42

## Performance

Cross-validation Accuracy:
92.52%




Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix


## Model Artifacts

Generated files:
models/

├── autism_model.pkl

└── encoders.pkl




## Conclusion

The developed machine learning pipeline successfully performs preprocessing, training, evaluation, and prediction for early autism detection.

The final Random Forest model achieved strong classification performance and can be integrated into an application through an API-based deployment.