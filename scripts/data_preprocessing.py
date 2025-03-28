import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """
    Handle missing values by filling with the mean of the column.
    """
    df.fillna(df.mean(), inplace=True)
    return df

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from the dataset using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables (e.g., CHAS) into numeric format if necessary.
    """
    if 'chas' in df.columns:
        df['chas'] = df['chas'].astype(int)
    return df

def scale_features(df, features):
    """
    Normalize or standardize numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path, target_column):
    """
    Complete data preprocessing pipeline.
    """
    print("Loading data...")
    df = load_data(file_path)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Removing outliers from key features...")
    for col in ['crim', 'zn', 'tax', 'lstat', 'rm']:
        df = remove_outliers(df, col)

    print("Encoding categorical variables...")
    df = encode_categorical_variables(df)

    print("Scaling numerical features...")
    numerical_features = ['crim', 'zn', 'indus', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    df = scale_features(df, numerical_features)

    print("Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    print("Preprocessing complete!")
    return X_train, X_test, y_train, y_test


file_path = r"C:/Users/SHYAM PANDIT/Omdena Assignment/assignment 6 (Machine Learning)/machine-learning-introduction-DhirajBro1/data/boston_housing.csv"
target_column = "medv"

 # Preprocess the data

X_train, X_test, y_train, y_test = preprocess_data(file_path, target_column)

# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

print("X_train shape:", X_train.shape)
print("X_train sample:\n", X_train.head())
print("y_train shape:", y_train.shape)
print("y_train sample:\n", y_train[:5])

print ("data preprocessed completed ")