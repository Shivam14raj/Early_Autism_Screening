import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from data_preprocessing import (
    load_data,
    clean_data,
    encode_features,
    save_encoders,
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


# Save encoders for future prediction
save_encoders(
    encoders,
    "models/encoders.pkl"
)



# Prepare X and y
X, y = prepare_data(df)



# -----------------------------
# 3. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------
# 4. Handle imbalance
# -----------------------------

smote = SMOTE(random_state=42)


X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)



# -----------------------------
# 5. Final Model
# -----------------------------

model = RandomForestClassifier(
    bootstrap=False,
    max_depth=20,
    n_estimators=50,
    random_state=42
)



# -----------------------------
# 6. Training
# -----------------------------

model.fit(
    X_train_smote,
    y_train_smote
)



# -----------------------------
# 7. Save Model
# -----------------------------

with open("models/autism_model.pkl", "wb") as f:

    pickle.dump(model, f)


print("Training completed successfully")
print("Model saved at models/autism_model.pkl")