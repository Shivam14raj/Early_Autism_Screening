import pickle
import pandas as pd


# -----------------------------
# Load model
# -----------------------------

with open("models/autism_model.pkl", "rb") as f:
    model = pickle.load(f)



# -----------------------------
# Load encoders
# -----------------------------

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)



# -----------------------------
# Prediction function
# -----------------------------

def predict(input_data):

    df = pd.DataFrame([input_data])


    # Apply same encoding used during training
    for col, encoder in encoders.items():

        df[col] = encoder.transform(df[col])


    prediction = model.predict(df)


    if prediction[0] == 1:
        return "ASD detected"
    
    else:
        return "No ASD detected"



# -----------------------------
# Test prediction
# -----------------------------

sample_data = {

    "A1_Score": 1,
    "A2_Score": 1,
    "A3_Score": 1,
    "A4_Score": 1,
    "A5_Score": 1,
    "A6_Score": 1,
    "A7_Score": 1,
    "A8_Score": 1,
    "A9_Score": 1,
    "A10_Score": 1,

    "age": 10,

    "gender": "m",

    "ethnicity": "White-European",

    "jaundice": "yes",

    "austim": "no",

    "contry_of_res": "United States",

    "used_app_before": "no",

    "result": 8,

    "relation": "Others"

}


result = predict(sample_data)

print(result)