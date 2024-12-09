import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the fraudTrain.csv file into a DataFrame
fraud_train_path = "C:/Users/mynam/Anmol/Code Projects/Machine learning/Credit Card/fraudTrain.csv"
df = pd.read_csv(fraud_train_path)

# Feature selection (using 'amt' as the only feature for simplicity in this example)
X = df[['amt']]  # Amount of the transaction
y = df['is_fraud']  # Fraud label (0 for not fraud, 1 for fraud)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy*100:.2f}%')

# Function to predict fraud based on transaction details
def is_fraud_transaction(amount, trans_date_time, cc_num):
    # Check if entered details match any record in the fraudTrain.csv
    match = df[(df['amt'] == amount) & (df['trans_date_trans_time'] == trans_date_time) & (df['cc_num'] == cc_num)]
    
    if match.empty:
        # If no match is found, flag as fraud
        return "Not Fraud"
    else:
        # If there is a match, check if the transaction is fraud based on the model prediction
        scaled_amount = scaler.transform([[amount]])
        prediction = model.predict(scaled_amount)
        
        if prediction == 1:
            return "Not Fraud"
        else:
            return "Fraud"

# Create the GUI window
root = tk.Tk()
root.title("Fraud Detection")

# Function to handle button click
def check_fraud():
    try:
        # Get transaction details from the user input
        trans_date_time = trans_date_time_entry.get()
        cc_num = cc_num_entry.get()
        amount = float(amount_entry.get())
        
        # Call the function to check if the transaction is fraud or not
        result = is_fraud_transaction(amount, trans_date_time, cc_num)
        
        # Show the result in a message box
        messagebox.showinfo("Fraud Detection Result", f'Transaction details:\nDate & Time: {trans_date_time}\nCredit Card: {cc_num}\nAmount: {amount}\nResult: {result}')
    except ValueError:
        # Show an error if the input is not valid
        messagebox.showerror("Invalid Input", "Please enter valid values for the transaction.")

# Create labels and entry widgets for transaction details
trans_date_time_label = tk.Label(root, text="Enter Transaction Date & Time (YYYY-MM-DD HH:MM:SS):")
trans_date_time_label.pack(padx=10, pady=5)

trans_date_time_entry = tk.Entry(root)
trans_date_time_entry.pack(padx=10, pady=5)

cc_num_label = tk.Label(root, text="Enter Credit Card Number:")
cc_num_label.pack(padx=10, pady=5)

cc_num_entry = tk.Entry(root)
cc_num_entry.pack(padx=10, pady=5)

amount_label = tk.Label(root, text="Enter Transaction Amount:")
amount_label.pack(padx=10, pady=5)

amount_entry = tk.Entry(root)
amount_entry.pack(padx=10, pady=5)

# Create a button to check for fraud
check_button = tk.Button(root, text="Check Fraud", command=check_fraud)
check_button.pack(padx=10, pady=20)

# Start the GUI event loop
root.mainloop()