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



# --------------------------------
# 1. Load Dataset
# --------------------------------

df = load_data("data/raw/train.csv")



# --------------------------------
# 2. Data Preprocessing
# --------------------------------

df = clean_data(df)



# Label Encoding

df, encoders = encode_features(df)



# Save encoders

save_encoders(
    encoders,
    "models/encoders.pkl"
)



# Prepare Features and Target

X, y = prepare_data(df)
print(y.value_counts())


# Save feature order
# Important for prediction time

with open("models/features.pkl", "wb") as f:

    pickle.dump(
        X.columns.tolist(),
        f
    )



# --------------------------------
# 3. Train Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)



# --------------------------------
# 4. Handle Class Imbalance
# --------------------------------

smote = SMOTE(
    random_state=42
)


X_train_smote, y_train_smote = smote.fit_resample(

    X_train,

    y_train

)



# --------------------------------
# 5. Initialize Final Model
# --------------------------------


model = RandomForestClassifier(

    bootstrap=False,

    max_depth=20,

    n_estimators=50,

    random_state=42

)



# --------------------------------
# 6. Train Model
# --------------------------------


model.fit(

    X_train_smote,

    y_train_smote

)



# --------------------------------
# 7. Save Model
# --------------------------------


with open("models/autism_model.pkl", "wb") as f:

    pickle.dump(

        model,

        f

    )



print("================================")
print("Training completed successfully")
print("Model saved: models/autism_model.pkl")
print("Encoders saved: models/encoders.pkl")
print("Features saved: models/features.pkl")
print("================================")