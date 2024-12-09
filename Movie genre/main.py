import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random
import os
import joblib

trained_model = None
tfidf_vectorizer = None
model_file = "movie_genre_model.pkl"
vectorizer_file = "movie_genre_vectorizer.pkl"

# data load
def load_train_data():
    """Loads the training data from train_data.txt"""
    data_path = "train_data.txt"
    train_data = []
    train_labels = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                train_labels.append(parts[2])
                train_data.append(parts[3])

    return train_data, train_labels

# model loader
def load_model():
    """Loads the trained model and vectorizer if they exist."""
    global trained_model, tfidf_vectorizer
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        trained_model = joblib.load(model_file)
        tfidf_vectorizer = joblib.load(vectorizer_file)

# training
def train_model(train_data, progress_callback=None):
    """Trains the model on the provided data."""
    data, labels = train_data

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data)

    model = LogisticRegression(max_iter=200)
    for i in range(10):
        model.fit(X, labels)
        if progress_callback:
            progress_callback(i + 1)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    return model, vectorizer

# training progress
def train_action():
    progress = tk.Toplevel()
    progress.title("Training Progress")
    tk.Label(progress, text="Training in Progress").pack()
    progress_bar = ttk.Progressbar(progress, length=300, mode='determinate')
    progress_bar.pack(pady=10)
    progress.update()

    def update_progress(step):
        progress_bar['value'] = step * 10
        progress.update()

    train_data = load_train_data()
    global trained_model, tfidf_vectorizer
    trained_model, tfidf_vectorizer = train_model(train_data, progress_callback=update_progress)

    progress.destroy()
    messagebox.showinfo("Training Complete", "Model has been trained 10 times!")

# model
def predict_genre(model, vectorizer, title, description):
    """Predicts the genre of a given movie description."""
    input_text = description
    X_input = vectorizer.transform([input_text])
    probabilities = model.predict_proba(X_input)[0]

    genre_probabilities = sorted(
        zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True
    )
    return genre_probabilities

# tester
def test_model_accuracy(model, vectorizer):
    """Tests the model on random 100 test samples and computes accuracy."""
    test_data_path = "test_data.txt"
    solution_path = "test_data_solution.txt"

    test_data = []
    test_labels = []

    with open(test_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                test_data.append(parts[2])

    with open(solution_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                test_labels.append(parts[2])

    indices = random.sample(range(len(test_data)), 100)
    sampled_data = [test_data[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]

    X_test = vectorizer.transform(sampled_data)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(sampled_labels, predictions)
    report = classification_report(sampled_labels, predictions, zero_division=0)

    return accuracy, report

# model results
def predict_action():
    if trained_model is None or tfidf_vectorizer is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    predict_window = tk.Toplevel()
    predict_window.title("Predict Movie Genre")

    tk.Label(predict_window, text="Enter Movie Title:").pack()
    title_entry = tk.Entry(predict_window, width=50)
    title_entry.pack()

    tk.Label(predict_window, text="Enter Movie Description:").pack()
    desc_entry = tk.Text(predict_window, height=10, width=50)
    desc_entry.pack()

    def predict_genre_action():
        title = title_entry.get().strip()
        description = desc_entry.get("1.0", tk.END).strip()
        if not description:
            messagebox.showerror("Error", "Description cannot be empty!")
            return

        genre_probabilities = predict_genre(trained_model, tfidf_vectorizer, title, description)
        result_window = tk.Toplevel()
        result_window.title("Prediction Results")

        tk.Label(result_window, text="Predicted Genre Probabilities:").pack()
        result_text = tk.Text(result_window, height=20, width=50)
        result_text.pack()

        for genre, probability in genre_probabilities:
            result_text.insert(tk.END, f"{genre}: {probability:.2%}\n")

        result_text.config(state=tk.DISABLED)

    tk.Button(predict_window, text="Predict", command=predict_genre_action).pack()


def test_accuracy_action():
    if trained_model is None or tfidf_vectorizer is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    accuracy, report = test_model_accuracy(trained_model, tfidf_vectorizer)

    result_window = tk.Toplevel()
    result_window.title("Model Accuracy")

    tk.Label(result_window, text=f"Accuracy: {accuracy:.2%}").pack()
    tk.Label(result_window, text="Detailed Report:").pack()
    result_text = tk.Text(result_window, height=20, width=80)
    result_text.pack()
    result_text.insert(tk.END, report)
    result_text.config(state=tk.DISABLED)


# Main GUI
def main():
    load_model() 

    root = tk.Tk()
    root.title("Movie Genre Classifier")

    tk.Label(root, text="Movie Genre Classification", font=("Arial", 16)).pack(pady=10)

    tk.Button(root, text="Train Model", command=train_action, width=25).pack(pady=5)
    tk.Button(root, text="Predict Genre", command=predict_action, width=25).pack(pady=5)
    tk.Button(root, text="Test Model Accuracy", command=test_accuracy_action, width=25).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()