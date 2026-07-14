import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


# Load dataset
def load_data(path):

    df = pd.read_csv(path)

    return df



# Cleaning data
def clean_data(df):

    # convert age datatype
    df['age'] = df['age'].astype(int)


    # drop unnecessary columns
    df = df.drop(columns=['ID', 'age_desc'])


    # country name correction
    mapping = {
        "Viet Nam": "Vietnam",
        "AmericanSamoa": "United States",
        "Hong Kong": "China"
    }

    df['contry_of_res'] = df['contry_of_res'].replace(mapping)


    # ethnicity cleaning
    df['ethnicity'] = df['ethnicity'].replace({
        '?': "Others",
        'others': "Others"
    })


    # relation cleaning
    df['relation'] = df['relation'].replace({
        "?": "Others",
        "Parent": "Others",
        "Relative": "Others",
        "Health care professional": "Others"
    })


    return df



# Label Encoding
def encode_features(df):

    # finding categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns


    encoders = {}


    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder


    return df, encoders



# Save encoders
def save_encoders(encoders, path):

    with open(path, "wb") as f:

        pickle.dump(encoders, f)



# Replace outliers using median
def replace_outliers_with_median(df, column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1


    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR


    median = df[column].median()


    df[column] = df[column].apply(
        lambda x: median if x < lower_bound or x > upper_bound else x
    )


    return df



# Prepare final X and y
def prepare_data(df):

    # replacing outliers
    df = replace_outliers_with_median(df, "age")

    df = replace_outliers_with_median(df, "result")


    # features
    X = df.drop(columns=["Class/ASD"])


    # target
    y = df["Class/ASD"]


    return X, y


if __name__ == "__main__":

    df = load_data("data/raw/train.csv")

    df = clean_data(df)

    df, encoders = encode_features(df)

    save_encoders(encoders, "models/encoders.pkl")

    X, y = prepare_data(df)

    print("Preprocessing completed")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape) 
    print("Encoded columns:", list(encoders.keys()))