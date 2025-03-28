# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import X_test, y_test  # Import preprocessed test data
from train_model import train_model  # Import the trained model function

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")

    # Predict the target variable using the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.7)
    plt.title("Residuals Distribution", fontsize=14)
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2)
    plt.title("Residuals vs Predicted Values", fontsize=14)
    plt.xlabel("Predicted Values", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print("Model evaluation completed!")

if __name__ == "__main__":
    # Train the model first (ensure model is trained)
    trained_model = train_model()

    # Evaluate the model
    evaluate_model(trained_model, X_test, y_test)
