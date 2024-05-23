import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np

class IrisClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Iris Flower Classifier")

        # Increase font size
        self.font = ('Arial', 14)
        self.title_font = ('Arial', 18, 'bold')
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabel", padding=5, font=self.font)
        self.style.configure("TButton", padding=5, font=self.font)
        self.style.configure("TFrame", padding=10)
        
        # Load the Iris dataset
        self.iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.iris.data, self.iris.target, test_size=0.2, random_state=42)
        self.classifier = RandomForestClassifier(random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

        self.create_widgets()
        self.plot_feature_importances()
        self.show_accuracy()
        self.create_scatter_plot()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.label = ttk.Label(self.main_frame, text="Enter the characteristics of the iris flower:", font=self.title_font)
        self.label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        self.sepal_length_label = ttk.Label(self.main_frame, text="Sepal Length:")
        self.sepal_length_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sepal_length_entry = ttk.Entry(self.main_frame, font=self.font)
        self.sepal_length_entry.grid(row=1, column=1, pady=5)

        self.sepal_width_label = ttk.Label(self.main_frame, text="Sepal Width:")
        self.sepal_width_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.sepal_width_entry = ttk.Entry(self.main_frame, font=self.font)
        self.sepal_width_entry.grid(row=2, column=1, pady=5)

        self.petal_length_label = ttk.Label(self.main_frame, text="Petal Length:")
        self.petal_length_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        self.petal_length_entry = ttk.Entry(self.main_frame, font=self.font)
        self.petal_length_entry.grid(row=3, column=1, pady=5)

        self.petal_width_label = ttk.Label(self.main_frame, text="Petal Width:")
        self.petal_width_label.grid(row=4, column=0, sticky=tk.W, pady=5)
        self.petal_width_entry = ttk.Entry(self.main_frame, font=self.font)
        self.petal_width_entry.grid(row=4, column=1, pady=5)

        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.predict_flower)
        self.predict_button.grid(row=5, column=0, pady=(20, 10))

        self.clear_button = ttk.Button(self.main_frame, text="Clear", command=self.clear_fields)
        self.clear_button.grid(row=5, column=1, pady=(20, 10))

        self.result_label = ttk.Label(self.main_frame, text="", font=self.font)
        self.result_label.grid(row=6, column=0, columnspan=3, pady=(10, 10))

        self.probabilities_label = ttk.Label(self.main_frame, text="", font=self.font)
        self.probabilities_label.grid(row=7, column=0, columnspan=3, pady=(10, 10))

        self.save_button = ttk.Button(self.main_frame, text="Save Model", command=self.save_model)
        self.save_button.grid(row=8, column=0, pady=(10, 10))

        self.load_button = ttk.Button(self.main_frame, text="Load Model", command=self.load_model)
        self.load_button.grid(row=8, column=1, pady=(10, 10))

        self.accuracy_label = ttk.Label(self.main_frame, text="", font=self.font)
        self.accuracy_label.grid(row=9, column=0, columnspan=3, pady=(10, 10))

    def validate_input(self, *args):
        try:
            float(self.sepal_length_entry.get())
            float(self.sepal_width_entry.get())
            float(self.petal_length_entry.get())
            float(self.petal_width_entry.get())
            return True
        except ValueError:
            return False

    def predict_flower(self):
        if not self.validate_input():
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return

        sepal_length = float(self.sepal_length_entry.get())
        sepal_width = float(self.sepal_width_entry.get())
        petal_length = float(self.petal_length_entry.get())
        petal_width = float(self.petal_width_entry.get())

        # Predict the type of iris flower
        prediction = self.classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        probabilities = self.classifier.predict_proba([[sepal_length, sepal_width, petal_length, petal_width]])

        # Convert numeric class to class name
        iris_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_flower = iris_names[prediction[0]]

        flower_colors = {0: 'Blue', 1: 'Red', 2: 'Yellow'}
        predicted_color = flower_colors[prediction[0]]

        self.result_label.config(text=f"Predicted flower type: {predicted_flower}\nPredicted flower color: {predicted_color}")

        probabilities_text = f"Setosa: {probabilities[0][0]:.2f}, Versicolor: {probabilities[0][1]:.2f}, Virginica: {probabilities[0][2]:.2f}"
        self.probabilities_label.config(text=f"Prediction Probabilities: {probabilities_text}")

    def clear_fields(self):
        self.sepal_length_entry.delete(0, tk.END)
        self.sepal_width_entry.delete(0, tk.END)
        self.petal_length_entry.delete(0, tk.END)
        self.petal_width_entry.delete(0, tk.END)
        self.result_label.config(text="")
        self.probabilities_label.config(text="")

    def plot_feature_importances(self):
        feature_importances = self.classifier.feature_importances_
        features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

        fig, ax = plt.subplots()
        ax.barh(features, feature_importances, color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=10, column=0, columnspan=3, pady=(10, 0))

    def show_accuracy(self):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            joblib.dump(self.classifier, file_path)
            messagebox.showinfo("Model Saved", f"Model saved to {file_path}")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.classifier = joblib.load(file_path)
            self.show_accuracy()
            self.plot_feature_importances()
            messagebox.showinfo("Model Loaded", f"Model loaded from {file_path}")

    def create_scatter_plot(self):
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.iris.data[:, 0], y=self.iris.data[:, 2], hue=self.iris.target, palette='viridis', ax=ax)
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Petal Length')
        ax.set_title('Iris Dataset - Sepal vs Petal Length')

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=11, column=0, columnspan=3, pady=(10, 0))

def main():
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
