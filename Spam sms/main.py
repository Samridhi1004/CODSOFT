import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Clean the text (remove non-alphabet characters, convert to lowercase, etc.)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    return " ".join(text.split())

# Preprocess the text (tokenize, remove stopwords, lemmatize)
def preprocess_text(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    # Remove stopwords from the tokens
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# Load and preprocess the dataset
def load_data():
    # Load the dataset
    data = pd.read_csv(r"C:\Users\mynam\Anmol\Code Projects\Machine learning\Spam sms\spam.csv", encoding="latin-1")
    
    # Drop unnecessary columns and rename others
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    data.rename(columns={"v1": "Target", "v2": "Text"}, inplace=True)
    
    # Clean the text
    data["Clean_Text"] = data["Text"].apply(clean_text)
    
    # Tokenize and lemmatize
    data["Processed_Text"] = data["Clean_Text"].apply(preprocess_text)
    
    # Create the corpus for TF-IDF
    corpus = [" ".join(words) for words in data["Processed_Text"]]
    
    # Vectorize the text using TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus).toarray()
    
    # Label encode the target variable
    data["Target"] = data["Target"].map({"legitimate": 0, "spam": 1})
    y = data["Target"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tfidf

# Train the model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Predict a new text message
def predict_message(message, model, tfidf):
    message = clean_text(message)
    processed_message = preprocess_text(message)
    vectorized_message = tfidf.transform([" ".join(processed_message)]).toarray()
    prediction = model.predict(vectorized_message)[0]
    probability = model.predict_proba(vectorized_message)[0][prediction]
    return "legitimate" if prediction == 0 else "spam", probability

# Show analytics (accuracy, confusion matrix)
def show_analytics(X_test, y_test, model):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    result = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    
    messagebox.showinfo("Model Analytics", result)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Spam"], yticklabels=["Legitimate", "Spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Main GUI
def main_gui():
    # Load data, preprocess, and train the model
    X_train, X_test, y_train, y_test, tfidf = load_data()
    model = train_model(X_train, y_train)
    
    # Create the main window
    window = tk.Tk()
    window.title("Spam Message Classifier")
    
    # Define actions for each button
    def on_test_model():
        message = simpledialog.askstring("Test the Model", "Enter the message:")
        if message:
            prediction, probability = predict_message(message, model, tfidf)
            messagebox.showinfo("Prediction", f"The message is {prediction} with {probability*100:.2f}% confidence.")
    
    def on_show_analytics():
        show_analytics(X_test, y_test, model)
    
    # Create buttons
    button_test = tk.Button(window, text="Test Model", command=on_test_model, width=30, height=2)
    button_test.pack(pady=20)
    
    button_analytics = tk.Button(window, text="Show Analytics", command=on_show_analytics, width=30, height=2)
    button_analytics.pack(pady=20)
    
    # Start the GUI
    window.mainloop()

if __name__ == "__main__":
    main_gui()
