import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

from data_preprocessing import (
    load_data,
    clean_data,
    encode_features,
    prepare_data
)


# -----------------------------
# 1. Load Data
# -----------------------------

df = load_data("data/raw/train.csv")


# -----------------------------
# 2. Preprocessing
# -----------------------------

df = clean_data(df)

df, encoders = encode_features(df)

X, y = prepare_data(df)



# -----------------------------
# 3. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------
# 4. Load Saved Model
# -----------------------------

with open("models/autism_model.pkl", "rb") as f:

    model = pickle.load(f)



# -----------------------------
# 5. Prediction
# -----------------------------

y_pred = model.predict(X_test)



# -----------------------------
# 6. Evaluation Metrics
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("Accuracy:")
print(accuracy)



print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)



# -----------------------------
# 7. Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(5,4))


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No ASD", "ASD"],
    yticklabels=["No ASD", "ASD"]
)


plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.title("Confusion Matrix")

plt.show()