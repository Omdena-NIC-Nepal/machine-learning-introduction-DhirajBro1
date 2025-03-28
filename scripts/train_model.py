# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import X_train, X_test, y_train, y_test  # Import preprocessed data

# Function to train the model
def train_model():
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    
    print("Model training complete!")

    return model  # Return the trained model

trained_model = train_model()
