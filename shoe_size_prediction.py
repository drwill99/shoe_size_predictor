"""
Author: Dallin Williams
Date: September 19, 2024

Program: shoe_size_prediction.py
Required Files: shoe-size-samples.csv

Function: This program uses a Linear Regression model to predict shoe size based on a user's height and sex. 
          It allows the user to input their height in feet and inches, and displays the predicted shoe size.
          The user can also check the accuracy of the prediction against their actual shoe size and plot their 
          shoe size on a scatterplot with dataset entries filtered by matching sex.

Notes:    The model will be more accurate with more data points. The dataset used for this program is limited.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt

# Load the data
file_path = 'shoe-size-samples.csv'  # Replace with actual path if needed
data = pd.read_csv(file_path)

# Filter out invalid shoe sizes and heights
data = data[(data['shoe_size'] >= 30) & (data['shoe_size'] <= 60)]
data = data[(data['height'] >= 100) & (data['height'] <= 250)]

# Drop rows with missing values in height or shoe_size
data = data.dropna(subset=['height', 'shoe_size'])

# Impute missing values in height with the median value (if needed)
imputer = SimpleImputer(strategy='median')
data['height'] = imputer.fit_transform(data[['height']])

# Preprocessing: Encoding the 'sex' column as 0 for male, 1 for female
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Selecting relevant columns
X = data[['height', 'sex']]
y = data['shoe_size']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Function to convert height from feet and inches to cm
def height_to_cm(feet, inches):
    return feet * 30.48 + inches * 2.54


# Function to convert European shoe sizes to American sizes based on sex
def convert_to_us_size(eu_size, sex):
    if sex == 'man':
        return eu_size - 33  # Conversion for men's shoes
    elif sex == 'woman':
        return eu_size - 30.5  # Conversion for women's shoes


# Predict shoe size based on height and sex
def predict_shoe_size(height_cm, sex):
    # Ensure the input is either 'man' or 'woman'
    if sex == 'm':
        sex = 'man'
    elif sex == 'f':
        sex = 'woman'
        
    sex_encoded = label_encoder.transform([sex])[0]  # 0 for man, 1 for woman

    # Create a DataFrame with proper column names for prediction
    input_data = pd.DataFrame([[height_cm, sex_encoded]], columns=['height', 'sex'])
    
    # Predict and convert the predicted European size to US size
    prediction = model.predict(input_data)
    us_size = convert_to_us_size(prediction[0], sex)
    return us_size


# Function to convert cm to inches
def cm_to_inches(cm):
    return cm / 2.54


# Function to calculate accuracy
def calculate_accuracy(predicted_size, actual_size):
    return 100 - abs((predicted_size - actual_size) / actual_size * 100)


# Function to plot and optionally save shoe sizes
def plot_shoe_size():
    try:
        actual_size = float(entry_actual_size.get())
        
        # Convert dataset European shoe sizes to American sizes and heights to inches
        data['us_shoe_size'] = data.apply(lambda row: convert_to_us_size(row['shoe_size'], 'man' if row['sex'] == 0 else 'woman'), axis=1)
        data['height_inches'] = data['height'].apply(cm_to_inches)

        # Filter the data to match the user's gender
        sex = entry_sex.get().lower()
        gender = 'man' if sex == 'm' else 'woman'
        filtered_data = data[data['sex'] == (0 if gender == 'man' else 1)]

        # Create plot with filtered data
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data['height_inches'], filtered_data['us_shoe_size'], label=f'{gender.capitalize()} Dataset Shoe Sizes', color='blue')

        # Plot user's shoe size with height in inches
        user_height_inches = cm_to_inches(height_to_cm(int(entry_feet.get()), int(entry_inches.get())))
        plt.scatter([user_height_inches], [actual_size], color='red', label="User's Shoe Size", s=100)

        # Labeling
        plt.title('Shoe Sizes (in US Sizes) vs. Height (in Inches)')
        plt.xlabel('Height (inches)')
        plt.ylabel('Shoe Size (US)')
        plt.legend()

        # Show plot
        plt.show()
    
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid shoe size.")


# Reset function to clear all input fields and labels
def reset():
    entry_feet.delete(0, tk.END)
    entry_inches.delete(0, tk.END)
    entry_sex.delete(0, tk.END)
    entry_actual_size.delete(0, tk.END)
    lbl_prediction.config(text="")
    lbl_accuracy.config(text="")


# GUI functions
def make_prediction():
    try:
        feet = int(entry_feet.get())
        inches = int(entry_inches.get())
        sex = entry_sex.get().lower()

        if sex not in ['m', 'f']:
            messagebox.showerror("Error", "Please enter 'm' or 'f' for sex.")
            return

        height_cm = height_to_cm(feet, inches)
        predicted_size = predict_shoe_size(height_cm, sex)

        lbl_prediction.config(text=f"Predicted shoe size (US): {predicted_size:.2f}")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for height.")


def check_accuracy():
    try:
        actual_size = float(entry_actual_size.get())
        predicted_size_text = lbl_prediction.cget("text")
        
        if "Predicted shoe size (US)" in predicted_size_text:
            predicted_size = float(predicted_size_text.split(": ")[1])
        else:
            messagebox.showerror("Error", "Please make a prediction first.")
            return

        accuracy = calculate_accuracy(predicted_size, actual_size)
        lbl_accuracy.config(text=f"Accuracy of the prediction: {accuracy:.2f}%")

    except ValueError:
        messagebox.showerror("Error", "Please enter a valid shoe size.")


# Creating the GUI using Tkinter
root = tk.Tk()
root.title("Shoe Size Predictor")

# Labels and input fields for height (feet and inches) and sex
tk.Label(root, text="Enter your height (feet):").grid(row=0, column=0)
entry_feet = tk.Entry(root)
entry_feet.grid(row=0, column=1)

tk.Label(root, text="Enter your height (inches):").grid(row=1, column=0)
entry_inches = tk.Entry(root)
entry_inches.grid(row=1, column=1)

tk.Label(root, text="Enter your sex (m/f):").grid(row=2, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=2, column=1)

# Button to make the prediction
btn_predict = tk.Button(root, text="Predict Shoe Size", command=make_prediction)
btn_predict.grid(row=3, column=1)

# Label to display the predicted shoe size
lbl_prediction = tk.Label(root, text="")
lbl_prediction.grid(row=4, column=1)

# Label and input for actual shoe size
tk.Label(root, text="Enter your actual shoe size (US):").grid(row=5, column=0)
entry_actual_size = tk.Entry(root)
entry_actual_size.grid(row=5, column=1)

# Button to check accuracy
btn_accuracy = tk.Button(root, text="Check Prediction Accuracy", command=check_accuracy)
btn_accuracy.grid(row=6, column=1)

# Button to plot the shoe size
btn_plot = tk.Button(root, text="Plot My Shoe Size", command=plot_shoe_size)
btn_plot.grid(row=7, column=1)

# Label to display the accuracy of the prediction
lbl_accuracy = tk.Label(root, text="")
lbl_accuracy.grid(row=8, column=1)

# Button to reset all fields
btn_reset = tk.Button(root, text="Reset", command=reset)
btn_reset.grid(row=9, column=1)

# Start the Tkinter event loop
root.mainloop()
